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
class MappedMemory(AutoFreePointer):
    """A memory pointer that refers to a buffer on the host that is mapped into
    device memory.

    :param context: The context in which the pointer was mapped.
    :type context: Context
    :param pointer: The address of the buffer.
    :type pointer: ctypes.c_void_p
    :param size: The size of the buffer in bytes.
    :type size: int
    :param owner: The owner is sometimes set by the internals of this class, or
                  used for Numba's internal memory management. It should not be
                  provided by an external user of the ``MappedMemory`` class
                  (e.g. from within an EMM Plugin); the default of `None`
                  should always suffice.
    :type owner: NoneType
    :param finalizer: A function that is called when the buffer is to be freed.
    :type finalizer: function
    """
    __cuda_memory__ = True

    def __init__(self, context, pointer, size, owner=None, finalizer=None):
        self.owned = owner
        self.host_pointer = pointer
        if USE_NV_BINDING:
            devptr = driver.cuMemHostGetDevicePointer(pointer, 0)
            self._bufptr_ = self.host_pointer
        else:
            devptr = drvapi.cu_device_ptr()
            driver.cuMemHostGetDevicePointer(byref(devptr), pointer, 0)
            self._bufptr_ = self.host_pointer.value
        self.device_pointer = devptr
        super(MappedMemory, self).__init__(context, devptr, size, finalizer=finalizer)
        self.handle = self.host_pointer
        self._buflen_ = self.size

    def own(self):
        return MappedOwnedPointer(weakref.proxy(self))