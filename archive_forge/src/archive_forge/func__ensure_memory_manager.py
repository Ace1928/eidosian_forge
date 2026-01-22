import sys
import os
import ctypes
import weakref
import functools
import warnings
import logging
import threading
import asyncio
import pathlib
from itertools import product
from abc import ABCMeta, abstractmethod
from ctypes import (c_int, byref, c_size_t, c_char, c_char_p, addressof,
import contextlib
import importlib
import numpy as np
from collections import namedtuple, deque
from numba import mviewbuf
from numba.core import utils, serialize, config
from .error import CudaSupportError, CudaDriverError
from .drvapi import API_PROTOTYPES
from .drvapi import cu_occupancy_b2d_size, cu_stream_callback_pyobj, cu_uuid
from numba.cuda.cudadrv import enums, drvapi, nvrtc, _extras
def _ensure_memory_manager():
    global _memory_manager
    if _memory_manager:
        return
    if config.CUDA_MEMORY_MANAGER == 'default':
        _memory_manager = NumbaCUDAMemoryManager
        return
    try:
        mgr_module = importlib.import_module(config.CUDA_MEMORY_MANAGER)
        set_memory_manager(mgr_module._numba_memory_manager)
    except Exception:
        raise RuntimeError('Failed to use memory manager from %s' % config.CUDA_MEMORY_MANAGER)