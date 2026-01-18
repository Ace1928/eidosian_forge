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
@staticmethod
def workloop(request_queue: Queue[Any], response_queue: Queue[Any]) -> None:
    """
        Work loop for the benchmarking subprocess.
        """
    while True:
        obj = request_queue.get()
        if obj is None:
            break
        elif isinstance(obj, Ping):
            response_queue.put(Pong())
        elif isinstance(obj, BenchmarkRequest):
            response_queue.put(obj.benchmark())
        else:
            raise RuntimeError(f'Invalid request type {type(obj)}')