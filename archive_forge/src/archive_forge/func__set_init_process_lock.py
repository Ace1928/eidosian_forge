import os
import sys
import warnings
from threading import RLock as threadRLock
from ctypes import CFUNCTYPE, c_int, CDLL, POINTER, c_uint
import numpy as np
import llvmlite.binding as ll
from llvmlite import ir
from numba.np.numpy_support import as_dtype
from numba.core import types, cgutils, config, errors
from numba.core.typing import signature
from numba.np.ufunc.wrappers import _wrapper_info
from numba.np.ufunc import ufuncbuilder
from numba.extending import overload, intrinsic
def _set_init_process_lock():
    global _backend_init_process_lock
    try:
        with _backend_init_thread_lock:
            import multiprocessing
            if 'fork' in multiprocessing.get_start_method() or _windows:
                ctx = multiprocessing.get_context()
                _backend_init_process_lock = ctx.RLock()
            else:
                _backend_init_process_lock = _nop()
    except OSError as e:
        msg = "Could not obtain multiprocessing lock due to OS level error: %s\nA likely cause of this problem is '/dev/shm' is missing or read-only such that necessary semaphores cannot be written.\n*** The responsibility of ensuring multiprocessing safe access to this initialization sequence/module import is deferred to the user! ***\n"
        warnings.warn(msg % str(e), errors.NumbaSystemWarning)
        _backend_init_process_lock = _nop()