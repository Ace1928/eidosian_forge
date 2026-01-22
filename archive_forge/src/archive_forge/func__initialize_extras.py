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
def _initialize_extras(self):
    if USE_NV_BINDING:
        return
    set_proto = ctypes.CFUNCTYPE(None, c_void_p)
    set_cuIpcOpenMemHandle = set_proto(_extras.set_cuIpcOpenMemHandle)
    set_cuIpcOpenMemHandle(self._find_api('cuIpcOpenMemHandle'))
    call_proto = ctypes.CFUNCTYPE(c_int, ctypes.POINTER(drvapi.cu_device_ptr), ctypes.POINTER(drvapi.cu_ipc_mem_handle), ctypes.c_uint)
    call_cuIpcOpenMemHandle = call_proto(_extras.call_cuIpcOpenMemHandle)
    call_cuIpcOpenMemHandle.__name__ = 'call_cuIpcOpenMemHandle'
    safe_call = self._ctypes_wrap_fn('call_cuIpcOpenMemHandle', call_cuIpcOpenMemHandle)
    self.cuIpcOpenMemHandle = safe_call