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
class IpcHandle(object):
    """
    CUDA IPC handle. Serialization of the CUDA IPC handle object is implemented
    here.

    :param base: A reference to the original allocation to keep it alive
    :type base: MemoryPointer
    :param handle: The CUDA IPC handle, as a ctypes array of bytes.
    :param size: Size of the original allocation
    :type size: int
    :param source_info: The identity of the device on which the IPC handle was
                        opened.
    :type source_info: dict
    :param offset: The offset into the underlying allocation of the memory
                   referred to by this IPC handle.
    :type offset: int
    """

    def __init__(self, base, handle, size, source_info=None, offset=0):
        self.base = base
        self.handle = handle
        self.size = size
        self.source_info = source_info
        self._impl = None
        self.offset = offset

    def _sentry_source_info(self):
        if self.source_info is None:
            raise RuntimeError("IPC handle doesn't have source info")

    def can_access_peer(self, context):
        """Returns a bool indicating whether the active context can peer
        access the IPC handle
        """
        self._sentry_source_info()
        if self.source_info == context.device.get_device_identity():
            return True
        source_device = Device.from_identity(self.source_info)
        return context.can_access_peer(source_device.id)

    def open_staged(self, context):
        """Open the IPC by allowing staging on the host memory first.
        """
        self._sentry_source_info()
        if self._impl is not None:
            raise ValueError('IpcHandle is already opened')
        self._impl = _StagedIpcImpl(self, self.source_info)
        return self._impl.open(context)

    def open_direct(self, context):
        """
        Import the IPC memory and returns a raw CUDA memory pointer object
        """
        if self._impl is not None:
            raise ValueError('IpcHandle is already opened')
        self._impl = _CudaIpcImpl(self)
        return self._impl.open(context)

    def open(self, context):
        """Open the IPC handle and import the memory for usage in the given
        context.  Returns a raw CUDA memory pointer object.

        This is enhanced over CUDA IPC that it will work regardless of whether
        the source device is peer-accessible by the destination device.
        If the devices are peer-accessible, it uses .open_direct().
        If the devices are not peer-accessible, it uses .open_staged().
        """
        if self.source_info is None or self.can_access_peer(context):
            fn = self.open_direct
        else:
            fn = self.open_staged
        return fn(context)

    def open_array(self, context, shape, dtype, strides=None):
        """
        Similar to `.open()` but returns an device array.
        """
        from . import devicearray
        if strides is None:
            strides = dtype.itemsize
        dptr = self.open(context)
        return devicearray.DeviceNDArray(shape=shape, strides=strides, dtype=dtype, gpu_data=dptr)

    def close(self):
        if self._impl is None:
            raise ValueError('IpcHandle not opened')
        self._impl.close()
        self._impl = None

    def __reduce__(self):
        if USE_NV_BINDING:
            preprocessed_handle = self.handle.reserved
        else:
            preprocessed_handle = tuple(self.handle)
        args = (self.__class__, preprocessed_handle, self.size, self.source_info, self.offset)
        return (serialize._rebuild_reduction, args)

    @classmethod
    def _rebuild(cls, handle_ary, size, source_info, offset):
        if USE_NV_BINDING:
            handle = binding.CUipcMemHandle()
            handle.reserved = handle_ary
        else:
            handle = drvapi.cu_ipc_mem_handle(*handle_ary)
        return cls(base=None, handle=handle, size=size, source_info=source_info, offset=offset)