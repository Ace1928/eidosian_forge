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
class _ActiveContext(object):
    """An contextmanager object to cache active context to reduce dependency
    on querying the CUDA driver API.

    Once entering the context, it is assumed that the active CUDA context is
    not changed until the context is exited.
    """
    _tls_cache = threading.local()

    def __enter__(self):
        is_top = False
        if hasattr(self._tls_cache, 'ctx_devnum'):
            hctx, devnum = self._tls_cache.ctx_devnum
        else:
            if USE_NV_BINDING:
                hctx = driver.cuCtxGetCurrent()
                if int(hctx) == 0:
                    hctx = None
            else:
                hctx = drvapi.cu_context(0)
                driver.cuCtxGetCurrent(byref(hctx))
                hctx = hctx if hctx.value else None
            if hctx is None:
                devnum = None
            else:
                if USE_NV_BINDING:
                    devnum = int(driver.cuCtxGetDevice())
                else:
                    hdevice = drvapi.cu_device()
                    driver.cuCtxGetDevice(byref(hdevice))
                    devnum = hdevice.value
                self._tls_cache.ctx_devnum = (hctx, devnum)
                is_top = True
        self._is_top = is_top
        self.context_handle = hctx
        self.devnum = devnum
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._is_top:
            delattr(self._tls_cache, 'ctx_devnum')

    def __bool__(self):
        """Returns True is there's a valid and active CUDA context.
        """
        return self.context_handle is not None
    __nonzero__ = __bool__