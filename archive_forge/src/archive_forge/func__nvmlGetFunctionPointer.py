from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def _nvmlGetFunctionPointer(name):
    global nvmlLib
    if name in _nvmlGetFunctionPointer_cache:
        return _nvmlGetFunctionPointer_cache[name]
    libLoadLock.acquire()
    try:
        if nvmlLib == None:
            raise NVMLError(NVML_ERROR_UNINITIALIZED)
        try:
            _nvmlGetFunctionPointer_cache[name] = getattr(nvmlLib, name)
            return _nvmlGetFunctionPointer_cache[name]
        except AttributeError:
            raise NVMLError(NVML_ERROR_FUNCTION_NOT_FOUND)
    finally:
        libLoadLock.release()