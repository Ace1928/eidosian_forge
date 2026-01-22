import logging
import os
import queue
import socket
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple
import torch.cuda.memory
import torch.cuda.nvtx
import torch.nn as nn
import torch.profiler
import torch.utils.hooks
class PyTorchProfiler:
    """Profiler which relies on native Pytorch profiling. Current setting of the profiler
    captures traces, memory footprint and other info that could be read via TensorBoard.
    """
    ACTIVITIES = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]

    def __init__(self, main_profiler: '_Profiler') -> None:
        self.main_profiler = main_profiler
        activities_str = '_'.join((a.name for a in self.ACTIVITIES))
        trace_handler = torch.profiler.tensorboard_trace_handler(dir_name=str(main_profiler.output_dir / f'profile_{activities_str}_{main_profiler.done_steps:06}'), worker_name=main_profiler.worker_name, use_gzip=True)
        self.hta = torch.profiler.profile(on_trace_ready=trace_handler, profile_memory=True, record_shapes=True, with_stack=True, activities=self.ACTIVITIES)
        self.done_steps = 0

    def __enter__(self):
        torch.cuda.synchronize()
        self.hta.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        self.hta.__exit__(exc_type, exc_val, exc_tb)

    def step(self) -> None:
        self.hta.step()
        self.done_steps += 1