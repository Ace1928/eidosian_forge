from itertools import product, cycle
import gc
import sys
import unittest
import warnings
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.core.errors import TypingError, NumbaValueError
from numba.np.numpy_support import as_dtype, numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, needs_blas
def test_sum_const_negative(self):
    self.disable_leak_check()

    @jit(nopython=True)
    def foo(arr):
        return arr.sum(axis=-3)
    a = np.ones((1, 2, 3, 4))
    self.assertPreciseEqual(foo(a), foo.py_func(a))
    a = np.ones((1, 2, 3))
    self.assertPreciseEqual(foo(a), foo.py_func(a))
    a = np.ones((1, 2))
    with self.assertRaises(NumbaValueError) as raises:
        foo(a)
    errmsg = "'axis' entry (-1) is out of bounds"
    self.assertIn(errmsg, str(raises.exception))
    with self.assertRaises(ValueError) as raises:
        foo.py_func(a)
    self.assertIn('out of bounds', str(raises.exception))