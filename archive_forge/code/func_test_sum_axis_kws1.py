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
def test_sum_axis_kws1(self):
    """ test sum with axis parameter over a whole range of dtypes  """
    pyfunc = array_sum_axis_kws
    cfunc = jit(nopython=True)(pyfunc)
    all_dtypes = [np.float64, np.float32, np.int64, np.uint64, np.complex64, np.complex128, TIMEDELTA_M]
    all_test_arrays = [[np.ones((7, 6, 5, 4, 3), arr_dtype), np.ones(1, arr_dtype), np.ones((7, 3), arr_dtype) * -5] for arr_dtype in all_dtypes]
    for arr_list in all_test_arrays:
        for arr in arr_list:
            for axis in (0, 1, 2):
                if axis > len(arr.shape) - 1:
                    continue
                with self.subTest('Testing np.sum(axis) with {} input '.format(arr.dtype)):
                    self.assertPreciseEqual(pyfunc(arr, axis=axis), cfunc(arr, axis=axis))