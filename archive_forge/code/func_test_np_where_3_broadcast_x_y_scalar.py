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
def test_np_where_3_broadcast_x_y_scalar(self):
    pyfunc = np_where_3
    cfunc = jit(nopython=True)(pyfunc)

    def check_ok(args):
        expected = pyfunc(*args)
        got = cfunc(*args)
        self.assertPreciseEqual(got, expected)

    def a_variations():
        a = np.linspace(-2, 4, 20)
        self.random.shuffle(a)
        yield a
        yield a.reshape(2, 5, 2)
        yield a.reshape(4, 5, order='F')
        yield a.reshape(2, 5, 2)[::-1]
    for a in a_variations():
        params = (a > 0, 0, 1)
        check_ok(params)
        params = (a < 0, np.nan, 1 + 4j)
        check_ok(params)
        params = (a > 1, True, False)
        check_ok(params)