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
def test_np_where_3_broadcast_x_or_y_scalar(self):
    pyfunc = np_where_3
    cfunc = jit(nopython=True)(pyfunc)

    def check_ok(args):
        condition, x, y = args
        expected = pyfunc(condition, x, y)
        got = cfunc(condition, x, y)
        self.assertPreciseEqual(got, expected)
        expected = pyfunc(condition, y, x)
        got = cfunc(condition, y, x)
        self.assertPreciseEqual(got, expected)

    def array_permutations():
        x = np.arange(9).reshape(3, 3)
        yield x
        yield (x * 1.1)
        yield np.asfortranarray(x)
        yield x[::-1]
        yield (np.linspace(-10, 10, 60).reshape(3, 4, 5) * 1j)

    def scalar_permutations():
        yield 0
        yield 4.3
        yield np.nan
        yield True
        yield (8 + 4j)
    for x in array_permutations():
        for y in scalar_permutations():
            x_mean = np.mean(x)
            condition = x > x_mean
            params = (condition, x, y)
            check_ok(params)