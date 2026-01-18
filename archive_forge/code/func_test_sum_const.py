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
def test_sum_const(self):
    pyfunc = array_sum_const_multi
    cfunc = jit(nopython=True)(pyfunc)
    arr = np.ones((3, 4, 5, 6, 7, 8))
    axis = 1
    self.assertPreciseEqual(pyfunc(arr, axis), cfunc(arr, axis))
    axis = 2
    self.assertPreciseEqual(pyfunc(arr, axis), cfunc(arr, axis))