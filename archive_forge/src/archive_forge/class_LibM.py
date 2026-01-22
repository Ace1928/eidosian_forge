import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
class LibM(types.WrapperAddressProtocol):

    def __init__(self, fname):
        if IS_WIN32:
            lib = ctypes.cdll.msvcrt
        else:
            libpath = ctypes.util.find_library('m')
            lib = ctypes.cdll.LoadLibrary(libpath)
        self.lib = lib
        self._name = fname
        if fname == 'cos':
            addr = ctypes.cast(self.lib.cos, ctypes.c_voidp).value
            signature = float64(float64)
        elif fname == 'sinf':
            addr = ctypes.cast(self.lib.sinf, ctypes.c_voidp).value
            signature = float32(float32)
        else:
            raise NotImplementedError(f'wrapper address of `{fname}` with signature `{signature}`')
        self._signature = signature
        self._address = addr

    def __repr__(self):
        return f'{type(self).__name__}({self._name!r})'

    def __wrapper_address__(self):
        return self._address

    def signature(self):
        return self._signature