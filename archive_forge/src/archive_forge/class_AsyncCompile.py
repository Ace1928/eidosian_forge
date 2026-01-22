from __future__ import annotations
import base64
import copyreg
import dataclasses
import functools
import hashlib
import importlib
import io
import json
import logging
import multiprocessing
import os
import pathlib
import pickle
import pkgutil
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
import threading
import warnings
import weakref
from bisect import bisect_right
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from copy import copy
from ctypes import c_void_p, cdll, CDLL
from dataclasses import field
from functools import partial
from importlib import abc
from pathlib import Path
from threading import Thread
from time import sleep, time
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING, Union
import torch
from torch._dynamo.device_interface import (
from torch._dynamo.utils import counters
from torch._inductor import config, exc
from torch._inductor.codegen.cuda import cuda_env
from torch._inductor.utils import cache_dir, developer_warning, is_linux
from torch._prims_common import suggest_memory_format
from torch.fx.experimental.symbolic_shapes import has_hint, hint_int, ShapeEnv
from torch.hub import _Faketqdm, tqdm
import torch
from ctypes import cdll
class AsyncCompile:

    def __init__(self) -> None:
        pass

    @staticmethod
    @functools.lru_cache(1)
    def pool() -> ThreadPoolExecutor:
        assert config.compile_threads > 1
        return ThreadPoolExecutor(config.compile_threads)

    @staticmethod
    @functools.lru_cache(1)
    def process_pool() -> ProcessPoolExecutor:
        caching_device_properties()
        assert config.compile_threads > 1
        orig_ppid = os.getpid()
        ctx = multiprocessing.get_context(config.worker_start_method)
        pool = ProcessPoolExecutor(config.compile_threads, mp_context=ctx, initializer=partial(_async_compile_initializer, orig_ppid))
        multiprocessing.util.Finalize(None, pool.shutdown, exitpriority=sys.maxsize)
        return pool

    @classmethod
    def warm_pool(cls) -> None:
        if config.compile_threads <= 1:
            return
        _compile_start()
        pool = cls.process_pool()
        if hasattr(pool, '_start_queue_management_thread'):
            pool._start_queue_management_thread()
        else:
            for _ in range(config.compile_threads):
                pool._adjust_process_count()
            if hasattr(pool, '_start_executor_manager_thread'):
                pool._start_executor_manager_thread()
        _compile_end()

    @classmethod
    def submit(cls, task: Callable[..., Any]) -> Any:
        if config.compile_threads <= 1:
            return task()
        return cls.pool().submit(task)

    @classmethod
    def map(cls, fn: Callable[..., Any], seq: List[Any]) -> List[Any]:
        if config.compile_threads <= 1 or len(seq) <= 1:
            return list(map(fn, seq))
        return [t.result() for t in [cls.pool().submit(fn, x) for x in seq]]

    def triton(self, kernel_name: str, source_code: str, device_str: str='cuda') -> Union[TritonFuture, ModuleType]:
        _compile_start()
        if config.compile_threads > 1:
            device_interface = get_interface_for_device(device_str)
            device = torch.device(device_str, device_interface.current_device())
            cc = device_interface.get_compute_capability(device)
            future = self.process_pool().submit(_worker_compile, kernel_name, source_code, cc, device)
            return TritonFuture(kernel_name, source_code, future)
        else:
            return _load_kernel(kernel_name, source_code)

    def cpp(self, source_code: str) -> ModuleType:

        def task():
            return CppCodeCache.load(source_code).kernel
        return self.submit(task)

    def cuda(self, source_code, dst_file_ext):

        def task():
            return CUDACodeCache.load(source_code, dst_file_ext)[0]
        return self.submit(task)

    def wait(self, scope: Dict[str, Any]) -> None:
        num_kernels = len([value for key, value in scope.items() if isinstance(value, (Future, TritonFuture))])
        pbar = tqdm(total=num_kernels, desc='Inductor Compilation', disable=config.disable_progress, delay=0)
        if config.compile_threads > 1:
            for key, result in scope.items():
                if config.verbose_progress and (not isinstance(pbar, _Faketqdm)):
                    pbar.set_postfix_str(key)
                if isinstance(result, (Future, TritonFuture)):
                    scope[key] = result.result()
                    pbar.update(1)
        _compile_end()