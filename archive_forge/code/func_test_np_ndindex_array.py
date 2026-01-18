import itertools
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
import unittest
def test_np_ndindex_array(self):
    func = np_ndindex_array
    arr = np.arange(12, dtype=np.int32) + 10
    self.check_array_unary(arr, typeof(arr), func)
    arr = arr.reshape((4, 3))
    self.check_array_unary(arr, typeof(arr), func)
    arr = arr.reshape((2, 2, 3))
    self.check_array_unary(arr, typeof(arr), func)