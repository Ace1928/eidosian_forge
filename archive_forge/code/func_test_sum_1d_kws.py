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
def test_sum_1d_kws(self):
    pyfunc = array_sum_axis_kws
    cfunc = jit(nopython=True)(pyfunc)
    a = np.arange(10.0)
    self.assertPreciseEqual(pyfunc(a, axis=0), cfunc(a, axis=0))
    pyfunc = array_sum_const_axis_neg_one
    cfunc = jit(nopython=True)(pyfunc)
    a = np.arange(10.0)
    self.assertPreciseEqual(pyfunc(a, axis=-1), cfunc(a, axis=-1))