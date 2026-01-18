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
def test_union1d_exceptions(self):
    cfunc = jit(nopython=True)(np_union1d)
    self.disable_leak_check()
    with self.assertRaises(TypingError) as raises:
        cfunc('Hello', np.array([1, 2]))
    self.assertIn('The arguments to np.union1d must be array-like', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc(np.array([1, 2]), 'Hello')
    self.assertIn('The arguments to np.union1d must be array-like', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc('Hello', 'World')
    self.assertIn('The arguments to np.union1d must be array-like', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc(np.array(['hello', 'world']), np.array(['a', 'b']))
    self.assertIn('For Unicode arrays, arrays must have same dtype', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc(np.array(['c', 'd']), np.array(['foo', 'bar']))
    self.assertIn('For Unicode arrays, arrays must have same dtype', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc(np.array(['c', 'd']), np.array([1, 2]))
    self.assertIn('For Unicode arrays, arrays must have same dtype', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc(np.array(['c', 'd']), np.array([1.1, 2.5]))
    self.assertIn('For Unicode arrays, arrays must have same dtype', str(raises.exception))