import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
def test_wrapper_address_protocol_libm(self):
    """Call cos and sinf from standard math library.

        """
    import ctypes.util

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
    mycos = LibM('cos')
    mysin = LibM('sinf')

    def myeval(f, x):
        return f(x)
    for jit_opts in [dict(nopython=True)]:
        jit_ = jit(**jit_opts)
        with self.subTest(jit=jit_opts):
            if mycos.signature() is not None:
                self.assertEqual(jit_(myeval)(mycos, 0.0), 1.0)
            if mysin.signature() is not None:
                self.assertEqual(jit_(myeval)(mysin, float32(0.0)), 0.0)