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
def test_np_frombuffer_dtype_non_const_str(self):

    @jit(nopython=True)
    def func(buf, dt):
        np.frombuffer(buf, dtype=dt)
    with self.assertRaises(TypingError) as raises:
        func(bytearray(range(16)), 'int32')
    excstr = str(raises.exception)
    msg = 'If np.frombuffer dtype is a string it must be a string constant.'
    self.assertIn(msg, excstr)