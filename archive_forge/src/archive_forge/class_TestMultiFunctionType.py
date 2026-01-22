import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
class TestMultiFunctionType(MemoryLeakMixin, TestCase):

    def test_base(self):
        nb_array = typeof(np.ones(2))
        callee_int_type = types.FunctionType(int64(int64))
        sig_int = int64(callee_int_type, int64)
        callee_array_type = types.FunctionType(float64(nb_array))
        sig_array = float64(callee_array_type, nb_array)

        @njit([sig_int, sig_array])
        def caller(callee, a):
            return callee(a)

        @njit
        def callee_int(b):
            return b

        @njit
        def callee_array(c):
            return c.sum()
        b = 1
        c = np.ones(2)
        self.assertEqual(caller(callee_int, b), b)
        self.assertEqual(caller(callee_array, c), c.sum())