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
def test_sum_axis_dtype_pos_arg(self):
    """ testing that axis and dtype inputs work when passed as positional """
    pyfunc = array_sum_axis_dtype_pos
    cfunc = jit(nopython=True)(pyfunc)
    dtype = np.float64
    a = np.ones((7, 6, 5, 4, 3))
    self.assertPreciseEqual(pyfunc(a, 1, dtype), cfunc(a, 1, dtype))
    self.assertPreciseEqual(pyfunc(a, 2, dtype), cfunc(a, 2, dtype))