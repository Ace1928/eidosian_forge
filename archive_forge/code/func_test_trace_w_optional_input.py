import contextlib
import gc
from itertools import product, cycle
import sys
import warnings
from numbers import Number, Integral
import platform
import numpy as np
from numba import jit, njit, typeof
from numba.core import errors
from numba.tests.support import (TestCase, tag, needs_lapack, needs_blas,
from .matmul_usecase import matmul_usecase
import unittest
def test_trace_w_optional_input(self):
    """Issue 2314"""

    @jit('(optional(float64[:,:]),)', nopython=True)
    def tested(a):
        return np.trace(a)
    a = np.ones((5, 5), dtype=np.float64)
    tested(a)
    with self.assertRaises(TypeError) as raises:
        tested(None)
    errmsg = str(raises.exception)
    self.assertEqual('expected array(float64, 2d, A), got None', errmsg)