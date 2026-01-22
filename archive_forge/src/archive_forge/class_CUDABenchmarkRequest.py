from __future__ import annotations
import contextlib
import dataclasses
import functools
import logging
import os
import queue
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from ctypes import byref, c_size_t, c_void_p
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from typing import (
import torch
from torch import multiprocessing
from torch._dynamo.testing import rand_strided
from torch._inductor import ir
from torch._inductor.codecache import CUDACodeCache, DLLWrapper, PyCodeCache
from . import config
from .utils import do_bench
from .virtualized import V
class CUDABenchmarkRequest(BenchmarkRequest):

    def __init__(self, kernel_name: str, input_tensor_meta: Union[TensorMeta, List[TensorMeta]], output_tensor_meta: Union[TensorMeta, List[TensorMeta]], extra_args: Iterable[Any], source_code: str):
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, extra_args)
        self.source_code = source_code
        self.workspace_size: int = 0
        self.workspace: Optional[torch.Tensor] = None
        self.DLL: Optional[DLLWrapper] = None
        self.hash_key: str = ''
        self.source_file: str = ''
        self.hash_key, self.source_file = CUDACodeCache.write(self.source_code, 'so')

    def make_run_fn(self, *input_tensors: torch.Tensor, output_tensor: torch.Tensor) -> Callable[[], None]:
        self.DLL, self.hash_key, self.source_file = CUDACodeCache.load(self.source_code, 'so')
        args = [c_void_p(tensor.data_ptr()) for tensor in list(input_tensors) + [output_tensor]]
        log.debug('make_run_fn: self.kernel_name=%s, self.source_file=%s, self.hash_key=%s, self.DLL=%s, args=%s, self.extra_args=%s', self.kernel_name, self.source_file, self.hash_key, self.DLL, args, self.extra_args)
        run_method = getattr(self.DLL, self.kernel_name)
        stream_ptr = c_void_p(torch.cuda.current_stream().cuda_stream)
        c_workspace_size = c_size_t()
        run_method(*args, *self.extra_args, byref(c_workspace_size), None, stream_ptr)
        self.workspace_size = c_workspace_size.value
        assert self.workspace_size == 0, 'Things need to be fixed to support non-zero workspace_size: 1) max autotune cache needs to store workspace size; 2) memory allocation needs to allocate / deallocate workspace correctly; '
        return functools.partial(run_method, *args, *self.extra_args, None, None, stream_ptr)

    def cleanup_run_fn(self) -> None:
        if self.DLL is not None:
            self.DLL.close()
        self.workspace = None

    def __str__(self) -> str:
        return f'self.kernel_name={self.kernel_name!r}, self.source_file={self.source_file!r}, self.hash_key={self.hash_key!r}'