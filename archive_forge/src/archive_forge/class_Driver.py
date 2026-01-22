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
class Driver(object):
    """
    Driver API functions are lazily bound.
    """
    _singleton = None

    def __new__(cls):
        obj = cls._singleton
        if obj is not None:
            return obj
        else:
            obj = object.__new__(cls)
            cls._singleton = obj
        return obj

    def __init__(self):
        self.devices = utils.UniqueDict()
        self.is_initialized = False
        self.initialization_error = None
        self.pid = None
        try:
            if config.DISABLE_CUDA:
                msg = 'CUDA is disabled due to setting NUMBA_DISABLE_CUDA=1 in the environment, or because CUDA is unsupported on 32-bit systems.'
                raise CudaSupportError(msg)
            self.lib = find_driver()
        except CudaSupportError as e:
            self.is_initialized = True
            self.initialization_error = e.msg

    def ensure_initialized(self):
        if self.is_initialized:
            return
        global _logger
        _logger = make_logger()
        self.is_initialized = True
        try:
            _logger.info('init')
            self.cuInit(0)
        except CudaAPIError as e:
            description = f'{e.msg} ({e.code})'
            self.initialization_error = description
            raise CudaSupportError(f'Error at driver init: {description}')
        else:
            self.pid = _getpid()
        self._initialize_extras()

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

    @property
    def is_available(self):
        self.ensure_initialized()
        return self.initialization_error is None

    def __getattr__(self, fname):
        self.ensure_initialized()
        if self.initialization_error is not None:
            raise CudaSupportError('Error at driver init: \n%s:' % self.initialization_error)
        if USE_NV_BINDING:
            return self._cuda_python_wrap_fn(fname)
        else:
            return self._ctypes_wrap_fn(fname)

    def _ctypes_wrap_fn(self, fname, libfn=None):
        if libfn is None:
            try:
                proto = API_PROTOTYPES[fname]
            except KeyError:
                raise AttributeError(fname)
            restype = proto[0]
            argtypes = proto[1:]
            libfn = self._find_api(fname)
            libfn.restype = restype
            libfn.argtypes = argtypes

        def verbose_cuda_api_call(*args):
            argstr = ', '.join([str(arg) for arg in args])
            _logger.debug('call driver api: %s(%s)', libfn.__name__, argstr)
            retcode = libfn(*args)
            self._check_ctypes_error(fname, retcode)

        def safe_cuda_api_call(*args):
            _logger.debug('call driver api: %s', libfn.__name__)
            retcode = libfn(*args)
            self._check_ctypes_error(fname, retcode)
        if config.CUDA_LOG_API_ARGS:
            wrapper = verbose_cuda_api_call
        else:
            wrapper = safe_cuda_api_call
        safe_call = functools.wraps(libfn)(wrapper)
        setattr(self, fname, safe_call)
        return safe_call

    def _cuda_python_wrap_fn(self, fname):
        libfn = getattr(binding, fname)

        def verbose_cuda_api_call(*args):
            argstr = ', '.join([str(arg) for arg in args])
            _logger.debug('call driver api: %s(%s)', libfn.__name__, argstr)
            return self._check_cuda_python_error(fname, libfn(*args))

        def safe_cuda_api_call(*args):
            _logger.debug('call driver api: %s', libfn.__name__)
            return self._check_cuda_python_error(fname, libfn(*args))
        if config.CUDA_LOG_API_ARGS:
            wrapper = verbose_cuda_api_call
        else:
            wrapper = safe_cuda_api_call
        safe_call = functools.wraps(libfn)(wrapper)
        setattr(self, fname, safe_call)
        return safe_call

    def _find_api(self, fname):
        if config.CUDA_PER_THREAD_DEFAULT_STREAM and (not USE_NV_BINDING):
            variants = ('_v2_ptds', '_v2_ptsz', '_ptds', '_ptsz', '_v2', '')
        else:
            variants = ('_v2', '')
        for variant in variants:
            try:
                return getattr(self.lib, f'{fname}{variant}')
            except AttributeError:
                pass

        def absent_function(*args, **kws):
            raise CudaDriverError(f'Driver missing function: {fname}')
        setattr(self, fname, absent_function)
        return absent_function

    def _detect_fork(self):
        if self.pid is not None and _getpid() != self.pid:
            msg = 'pid %s forked from pid %s after CUDA driver init'
            _logger.critical(msg, _getpid(), self.pid)
            raise CudaDriverError('CUDA initialized before forking')

    def _check_ctypes_error(self, fname, retcode):
        if retcode != enums.CUDA_SUCCESS:
            errname = ERROR_MAP.get(retcode, 'UNKNOWN_CUDA_ERROR')
            msg = 'Call to %s results in %s' % (fname, errname)
            _logger.error(msg)
            if retcode == enums.CUDA_ERROR_NOT_INITIALIZED:
                self._detect_fork()
            raise CudaAPIError(retcode, msg)

    def _check_cuda_python_error(self, fname, returned):
        retcode = returned[0]
        retval = returned[1:]
        if len(retval) == 1:
            retval = retval[0]
        if retcode != binding.CUresult.CUDA_SUCCESS:
            msg = 'Call to %s results in %s' % (fname, retcode.name)
            _logger.error(msg)
            if retcode == binding.CUresult.CUDA_ERROR_NOT_INITIALIZED:
                self._detect_fork()
            raise CudaAPIError(retcode, msg)
        return retval

    def get_device(self, devnum=0):
        dev = self.devices.get(devnum)
        if dev is None:
            dev = Device(devnum)
            self.devices[devnum] = dev
        return weakref.proxy(dev)

    def get_device_count(self):
        if USE_NV_BINDING:
            return self.cuDeviceGetCount()
        count = c_int()
        self.cuDeviceGetCount(byref(count))
        return count.value

    def list_devices(self):
        """Returns a list of active devices
        """
        return list(self.devices.values())

    def reset(self):
        """Reset all devices
        """
        for dev in self.devices.values():
            dev.reset()

    def pop_active_context(self):
        """Pop the active CUDA context and return the handle.
        If no CUDA context is active, return None.
        """
        with self.get_active_context() as ac:
            if ac.devnum is not None:
                if USE_NV_BINDING:
                    return driver.cuCtxPopCurrent()
                else:
                    popped = drvapi.cu_context()
                    driver.cuCtxPopCurrent(byref(popped))
                    return popped

    def get_active_context(self):
        """Returns an instance of ``_ActiveContext``.
        """
        return _ActiveContext()

    def get_version(self):
        """
        Returns the CUDA Runtime version as a tuple (major, minor).
        """
        if USE_NV_BINDING:
            version = driver.cuDriverGetVersion()
        else:
            dv = ctypes.c_int(0)
            driver.cuDriverGetVersion(ctypes.byref(dv))
            version = dv.value
        major = version // 1000
        minor = (version - major * 1000) // 10
        return (major, minor)