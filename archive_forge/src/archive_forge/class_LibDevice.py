import logging
import re
import sys
import warnings
from ctypes import (c_void_p, c_int, POINTER, c_char_p, c_size_t, byref,
import threading
from llvmlite import ir
from .error import NvvmError, NvvmSupportError, NvvmWarning
from .libs import get_libdevice, open_libdevice, open_cudalib
from numba.core import cgutils, config
class LibDevice(object):
    _cache_ = None

    def __init__(self):
        if self._cache_ is None:
            if get_libdevice() is None:
                raise RuntimeError(MISSING_LIBDEVICE_FILE_MSG)
            self._cache_ = open_libdevice()
        self.bc = self._cache_

    def get(self):
        return self.bc