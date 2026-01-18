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
def test_sum_axis_kws2(self):
    """  testing uint32 and int32 separately

        uint32 and int32 must be tested separately because Numpy's current
        behaviour is different in 64bits Windows (accumulates as int32)
        and 64bits Linux (accumulates as int64), while Numba has decided to always
        accumulate as int64, when the OS is 64bits. No testing has been done
        for behaviours in 32 bits platforms.
        """
    pyfunc = array_sum_axis_kws
    cfunc = jit(nopython=True)(pyfunc)
    all_dtypes = [np.int32, np.uint32]
    out_dtypes = {np.dtype('int32'): np.int64, np.dtype('uint32'): np.uint64, np.dtype('int64'): np.int64, np.dtype(TIMEDELTA_M): np.dtype(TIMEDELTA_M)}
    all_test_arrays = [[np.ones((7, 6, 5, 4, 3), arr_dtype), np.ones(1, arr_dtype), np.ones((7, 3), arr_dtype) * -5] for arr_dtype in all_dtypes]
    for arr_list in all_test_arrays:
        for arr in arr_list:
            for axis in (0, 1, 2):
                if axis > len(arr.shape) - 1:
                    continue
                with self.subTest('Testing np.sum(axis) with {} input '.format(arr.dtype)):
                    npy_res = pyfunc(arr, axis=axis)
                    numba_res = cfunc(arr, axis=axis)
                    if isinstance(numba_res, np.ndarray):
                        self.assertPreciseEqual(npy_res.astype(out_dtypes[arr.dtype]), numba_res.astype(out_dtypes[arr.dtype]))
                    else:
                        self.assertEqual(npy_res, numba_res)