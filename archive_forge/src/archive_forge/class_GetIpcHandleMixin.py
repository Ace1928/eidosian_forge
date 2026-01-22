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
class GetIpcHandleMixin:
    """A class that provides a default implementation of ``get_ipc_handle()``.
    """

    def get_ipc_handle(self, memory):
        """Open an IPC memory handle by using ``cuMemGetAddressRange`` to
        determine the base pointer of the allocation. An IPC handle of type
        ``cu_ipc_mem_handle`` is constructed and initialized with
        ``cuIpcGetMemHandle``. A :class:`numba.cuda.IpcHandle` is returned,
        populated with the underlying ``ipc_mem_handle``.
        """
        base, end = device_extents(memory)
        if USE_NV_BINDING:
            ipchandle = driver.cuIpcGetMemHandle(base)
            offset = int(memory.handle) - int(base)
        else:
            ipchandle = drvapi.cu_ipc_mem_handle()
            driver.cuIpcGetMemHandle(byref(ipchandle), base)
            offset = memory.handle.value - base
        source_info = self.context.device.get_device_identity()
        return IpcHandle(memory, ipchandle, memory.size, source_info, offset=offset)