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
class NumbaCUDAMemoryManager(GetIpcHandleMixin, HostOnlyCUDAMemoryManager):
    """Internal on-device memory management for Numba. This is implemented using
    the EMM Plugin interface, but is not part of the public API."""

    def initialize(self):
        if self.deallocations.memory_capacity == _SizeNotSet:
            self.deallocations.memory_capacity = self.get_memory_info().total

    def memalloc(self, size):
        if USE_NV_BINDING:

            def allocator():
                return driver.cuMemAlloc(size)
            ptr = self._attempt_allocation(allocator)
            alloc_key = ptr
        else:
            ptr = drvapi.cu_device_ptr()

            def allocator():
                driver.cuMemAlloc(byref(ptr), size)
            self._attempt_allocation(allocator)
            alloc_key = ptr.value
        finalizer = _alloc_finalizer(self, ptr, alloc_key, size)
        ctx = weakref.proxy(self.context)
        mem = AutoFreePointer(ctx, ptr, size, finalizer=finalizer)
        self.allocations[alloc_key] = mem
        return mem.own()

    def get_memory_info(self):
        if USE_NV_BINDING:
            free, total = driver.cuMemGetInfo()
        else:
            free = c_size_t()
            total = c_size_t()
            driver.cuMemGetInfo(byref(free), byref(total))
            free = free.value
            total = total.value
        return MemoryInfo(free=free, total=total)

    @property
    def interface_version(self):
        return _SUPPORTED_EMM_INTERFACE_VERSION