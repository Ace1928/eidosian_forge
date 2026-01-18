import itertools
import math
import platform
from functools import partial
from itertools import product
import warnings
from textwrap import dedent
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.typed import List, Dict
from numba.np.numpy_support import numpy_version
from numba.core.errors import TypingError, NumbaDeprecationWarning
from numba.core.config import IS_32BITS
from numba.core.utils import pysignature
from numba.np.extensions import cross2d
from numba.tests.support import (TestCase, MemoryLeakMixin,
import unittest
def test_take_along_axis_exceptions(self):
    arr2d = np.arange(8).reshape(2, 4)
    indices_none = np.array([0, 1], dtype=np.uint64)
    indices = np.ones((2, 4), dtype=np.uint64)

    def gen(axis):

        @njit
        def impl(a, i):
            return np.take_along_axis(a, i, axis)
        return impl
    with self.assertRaises(TypingError) as raises:
        gen('a')(arr2d, indices)
    self.assertIn('axis must be an integer', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        gen(-3)(arr2d, indices)
    self.assertIn('axis is out of bounds', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        gen(2)(arr2d, indices)
    self.assertIn('axis is out of bounds', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        gen(None)(12, indices_none)
    self.assertIn('"arr" must be an array', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        gen(None)(arr2d, 5)
    self.assertIn('"indices" must be an array', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        gen(None)(arr2d, np.array([0.0, 1.0]))
    self.assertIn('indices array must contain integers', str(raises.exception))

    @njit
    def not_literal_axis(a, i, axis):
        return np.take_along_axis(a, i, axis)
    with self.assertRaises(TypingError) as raises:
        not_literal_axis(arr2d, indices, 0)
    self.assertIn('axis must be a literal value', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        gen(0)(arr2d, np.array([0, 1], dtype=np.uint64))
    self.assertIn('must have the same number of dimensions', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        gen(None)(arr2d, arr2d)
    self.assertIn('must have the same number of dimensions', str(raises.exception))
    with self.assertRaises(ValueError) as raises:
        gen(0)(arr2d, np.ones((2, 3), dtype=np.uint64))
    self.assertIn("dimensions don't match", str(raises.exception))
    self.disable_leak_check()