# coding=utf-8
import os
import sys
import time
from pathlib import Path
from typing import Optional, List
import threading
from queue import Queue

# 获取项目根目录
if getattr(sys, 'frozen', False):
    PROJ_DIR = Path(sys.executable)
else:
    PROJ_DIR = Path(__file__).parent  # 假设 server.py 在子目录中

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from qwen_asr_gguf.inference import QwenASREngine, ASREngineConfig, AlignerConfig, exporters

# 创建 app 对象
app = typer.Typer(help="Qwen3-ASR 后台服务", add_completion=False)
console = Console()

def get_model_filenames(precision: str, is_aligner: bool = False):
    """根据精度返回对应的模型文件名"""
    prefix = "qwen3_aligner" if is_aligner else "qwen3_asr"
    return {
        "frontend": f"{prefix}_encoder_frontend.{precision}.onnx",
        "backend": f"{prefix}_encoder_backend.{precision}.onnx"
    }

def check_model_files(config: ASREngineConfig):
    """检查模型文件完整性"""
    missing_files = []
    
    # ASR 核心文件
    asr_llm = Path(config.model_dir) / config.llm_fn
    asr_frontend = Path(config.model_dir) / config.encoder_frontend_fn
    asr_backend = Path(config.model_dir) / config.encoder_backend_fn
    
    for f in [asr_llm, asr_frontend, asr_backend]:
        if not f.exists():
            missing_files.append(str(f))
            
    # Aligner 文件
    if config.enable_aligner and config.align_config:
        align_llm = Path(config.align_config.model_dir) / config.align_config.llm_fn
        align_frontend = Path(config.align_config.model_dir) / config.align_config.encoder_frontend_fn
        align_backend = Path(config.align_config.model_dir) / config.align_config.encoder_backend_fn
        
        for f in [align_llm, align_frontend, align_backend]:
            if not f.exists():
                missing_files.append(str(f))
    
    if missing_files:
        console.print("\n[bold red]错误：找不到以下所需模型文件：[/bold red]")
        for f in missing_files:
            console.print(f"  - {f}")
        console.print("\n[bold yellow]请到以下链接下载模型文件，并解压到 model 目录：[/bold yellow]")
        console.print("[blue]https://github.com/HaujetZhao/Qwen3-ASR-GGUF/releases/tag/models[/blue]\n")
        raise typer.Exit(code=1)

class AudioFileHandler:
    """处理音频文件的类"""
    def __init__(self, engine, output_dir=None, supported_exts=None):
        self.engine = engine
        self.output_dir = Path(output_dir) if output_dir else None
        self.supported_exts = supported_exts or {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.m4a'}
        self.processing = set()
        self.queue = Queue()
        self.running = True
        
    def add_file(self, file_path: Path):
        """添加文件到处理队列"""
        if file_path.suffix.lower() in self.supported_exts:
            if str(file_path) not in self.processing:
                self.processing.add(str(file_path))
                self.queue.put(file_path)
                console.print(f"[cyan]已添加到队列: {file_path.name}[/cyan]")
                
    def process_worker(self, language=None, context=None, temperature=0.4):
        """后台处理工作线程"""
        while self.running:
            try:
                audio_path = self.queue.get(timeout=1)
            except:
                continue
                
            try:
                console.print(f"\n[bold blue]开始处理:[/bold blue] {audio_path.name}")
                
                # 确定输出路径
                if self.output_dir:
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                    base_out = self.output_dir / audio_path.stem
                else:
                    base_out = audio_path.with_suffix("")
                
                txt_out = f"{base_out}.txt"
                
                # 转录
                res = self.engine.transcribe(
                    audio_file=str(audio_path),
                    language=language,
                    context=context,
                    temperature=temperature,
                    start_second=0,
                    duration=None
                )

                # 导出结果
                exporters.export_to_txt(txt_out, res)
                console.print(f"[green]✓ 转录完成: {txt_out}[/green]")

                if self.engine.config.enable_aligner and res.alignment:
                    srt_out = f"{base_out}.srt"
                    json_out = f"{base_out}.json"
                    exporters.export_to_srt(srt_out, res)
                    exporters.export_to_json(json_out, res)
                    
            except Exception as e:
                console.print(f"[red]处理失败 {audio_path.name}: {e}[/red]")
            finally:
                self.processing.discard(str(audio_path))
                self.queue.task_done()
    
    def shutdown(self):
        """关闭处理器"""
        self.running = False

class DirectoryWatcher:
    """目录监控器"""
    def __init__(self, handler, watch_dir):
        self.handler = handler
        self.watch_dir = Path(watch_dir)
        self.running = True
        
    def start(self):
        """启动监控（简单轮询方式，无需额外依赖）"""
        console.print(f"[green]开始监控目录: {self.watch_dir}[/green]")
        
        # 记录已处理的文件
        processed_files = set()
        
        while self.running:
            try:
                # 扫描目录中的音频文件
                for ext in self.handler.supported_exts:
                    for file_path in self.watch_dir.glob(f"*{ext}"):
                        if str(file_path) not in processed_files:
                            # 等待文件写入完成
                            time.sleep(0.5)
                            self.handler.add_file(file_path)
                            processed_files.add(str(file_path))
                
                time.sleep(2)  # 每2秒扫描一次
            except Exception as e:
                console.print(f"[yellow]监控错误: {e}[/yellow]")
                time.sleep(5)
    
    def stop(self):
        """停止监控"""
        self.running = False

@app.command()
def start(
    watch_dir: Path = typer.Argument(..., help="要监控的目录路径"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="输出目录（默认与输入文件同目录）"),
    
    # 模型配置
    model_dir: str = typer.Option(str(PROJ_DIR / "model"), "--model-dir", "-m"),
    precision: str = typer.Option("int4", "--prec", help="编码器精度: fp32, fp16, int8, int4"),
    timestamp: bool = typer.Option(True, "--timestamp/--no-ts", help="是否开启时间戳引擎"),
    onnx_provider: str = typer.Option("DML", "--provider", help="ONNX 执行后端: CPU, CUDA, DML"),
    llm_use_gpu: bool = typer.Option(True, "--gpu/--no-gpu", help="LLM 是否使用 GPU 加速"),
    use_vulkan: bool = typer.Option(True, "--vulkan/--no-vulkan", help="是否开启 Vulkan 加速"),
    n_ctx: int = typer.Option(2048, "--n-ctx", help="LLM 上下文窗口大小"),
    
    # 转录设置
    language: Optional[str] = typer.Option(None, "--language", "-l", help="强制指定语种"),
    context: str = typer.Option("", "--context", "-p", help="上下文提示词"),
    temperature: float = typer.Option(0.4, "--temperature", help="采样温度"),
    
    # 性能配置
    chunk_size: float = typer.Option(40.0, "--chunk-size", help="分段识别时长"),
    memory_num: int = typer.Option(1, "--memory-num", help="记忆的历史片段数量"),
    
    # 其他
    verbose: bool = typer.Option(False, "--verbose/--quiet", help="是否打印详细日志"),
):
    """启动后台服务，监控目录并自动转录音频文件"""
    
    # 环境准备
    if not use_vulkan:
        os.environ["VK_ICD_FILENAMES"] = "none"
    
    # 创建输出目录
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 构造配置
    asr_files = get_model_filenames(precision, is_aligner=False)
    align_files = get_model_filenames(precision, is_aligner=True)
    
    align_config = None
    if timestamp:
        align_config = AlignerConfig(
            model_dir=model_dir,
            onnx_provider=onnx_provider,
            llm_use_gpu=llm_use_gpu,
            encoder_frontend_fn=align_files["frontend"],
            encoder_backend_fn=align_files["backend"],
            n_ctx=n_ctx
        )
    
    config = ASREngineConfig(
        model_dir=model_dir,
        onnx_provider=onnx_provider,
        llm_use_gpu=llm_use_gpu,
        encoder_frontend_fn=asr_files["frontend"],
        encoder_backend_fn=asr_files["backend"],
        n_ctx=n_ctx,
        chunk_size=chunk_size,
        memory_num=memory_num,
        enable_aligner=timestamp,
        align_config=align_config,
        verbose=verbose
    )
    
    # 打印配置
    config_table = Table(show_header=False, box=None)
    config_table.add_row("监控目录", f"[green]{watch_dir}[/green]")
    config_table.add_row("输出目录", f"[green]{output_dir or '与源文件同目录'}[/green]")
    config_table.add_row("模型目录", f"[green]{model_dir}[/green]")
    config_table.add_row("编码精度", f"[cyan]{precision}[/cyan]")
    config_table.add_row("加速设备", f"ONNX:{onnx_provider} | LLM-GPU:{'ON' if llm_use_gpu else 'OFF'} | Vulkan:{'ON' if use_vulkan else 'OFF'}")
    config_table.add_row("语言设定", f"{language or '自动识别'}")
    
    console.print(Panel(config_table, title="[bold cyan]Qwen3-ASR 后台服务配置[/bold cyan]", expand=False))
    
    # 检查模型文件
    check_model_files(config)
    
    # 检查监控目录
    if not watch_dir.exists():
        console.print(f"[red]错误：监控目录不存在: {watch_dir}[/red]")
        raise typer.Exit(code=1)
    
    # 初始化引擎
    console.print("[bold yellow]正在初始化引擎，请稍候...[/bold yellow]")
    try:
        t0 = time.time()
        engine = QwenASREngine(config=config)
        init_duration = time.time() - t0
        console.print(f"[green]✓ 引擎初始化完成，耗时: {init_duration:.2f} 秒[/green]")
    except Exception as e:
        console.print(f"[bold red]引擎初始化失败:[/bold red]\n{e}")
        raise typer.Exit(code=1)
    
    # 创建文件处理器
    handler = AudioFileHandler(engine, output_dir=output_dir)
    
    # 启动处理线程
    process_thread = threading.Thread(
        target=handler.process_worker,
        args=(language, context, temperature),
        daemon=True
    )
    process_thread.start()
    
    # 启动目录监控
    watcher = DirectoryWatcher(handler, watch_dir)
    
    try:
        watcher.start()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]正在停止服务...[/bold yellow]")
        handler.shutdown()
        watcher.stop()
        engine.shutdown()
        console.print("[bold green]服务已停止[/bold green]")

@app.command()
def process(
    files: List[Path] = typer.Argument(..., help="要转录的音频文件列表"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="输出目录"),
    
    # 模型配置（复用上面的参数）
    model_dir: str = typer.Option(str(PROJ_DIR / "model"), "--model-dir", "-m"),
    precision: str = typer.Option("int4", "--prec"),
    timestamp: bool = typer.Option(True, "--timestamp/--no-ts"),
    onnx_provider: str = typer.Option("DML", "--provider"),
    llm_use_gpu: bool = typer.Option(True, "--gpu/--no-gpu"),
    use_vulkan: bool = typer.Option(True, "--vulkan/--no-vulkan"),
    n_ctx: int = typer.Option(2048, "--n-ctx"),
    language: Optional[str] = typer.Option(None, "--language", "-l"),
    context: str = typer.Option("", "--context", "-p"),
    temperature: float = typer.Option(0.4, "--temperature"),
    chunk_size: float = typer.Option(40.0, "--chunk-size"),
    memory_num: int = typer.Option(1, "--memory-num"),
    verbose: bool = typer.Option(False, "--verbose/--quiet"),
    yes: bool = typer.Option(False, "--yes", "-y", help="覆盖已存在的输出文件"),
):
    """处理单个或多个音频文件（一次性任务）"""
    
    # 这个函数可以复用原来的 transcribe 逻辑
    # 为了简洁，这里省略，可以复制原来的 transcribe 函数内容
    
    console.print("[yellow]请使用原始脚本的 transcribe 命令处理单个文件[/yellow]")
    console.print("示例: python your_script.py transcribe audio.wav")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    app()