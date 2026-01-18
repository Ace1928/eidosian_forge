from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_numpy_resize_exception(self):
    self.disable_leak_check()
    cfunc = njit(numpy_resize)
    with self.assertRaises(TypingError) as raises:
        cfunc('abc', (2, 3))
    self.assertIn('The argument "a" must be array-like', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc(np.array([[0, 1], [2, 3]]), 'abc')
    self.assertIn('The argument "new_shape" must be an integer or a tuple of integers', str(raises.exception))
    with self.assertRaises(ValueError) as raises:
        cfunc(np.array([[0, 1], [2, 3]]), (-2, 3))
    self.assertIn('All elements of `new_shape` must be non-negative', str(raises.exception))
    with self.assertRaises(ValueError) as raises:
        cfunc(np.array([[0, 1], [2, 3]]), -4)
    self.assertIn('All elements of `new_shape` must be non-negative', str(raises.exception))