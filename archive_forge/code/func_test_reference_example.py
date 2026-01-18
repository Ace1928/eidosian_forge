import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
def test_reference_example(self):
    import numba

    @numba.njit
    def composition(funcs, x):
        r = x
        for f in funcs[::-1]:
            r = f(r)
        return r

    @numba.cfunc('double(double)')
    def a(x):
        return x + 1.0

    @numba.njit()
    def b(x):
        return x * x
    r = composition((a, b, b, a), 0.5)
    self.assertEqual(r, (0.5 + 1.0) ** 4 + 1.0)
    r = composition((b, a, b, b, a), 0.5)
    self.assertEqual(r, ((0.5 + 1.0) ** 4 + 1.0) ** 2)