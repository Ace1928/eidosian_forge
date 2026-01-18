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
def test_swapaxes_exception(self):
    pyfunc = swapaxes
    cfunc = jit(nopython=True)(pyfunc)
    self.disable_leak_check()
    with self.assertRaises(TypingError) as raises:
        cfunc('abc', 0, 0)
    self.assertIn('The first argument "a" must be an array', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc(np.arange(4), 'abc', 0)
    self.assertIn('The second argument "axis1" must be an integer', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc(np.arange(4), 0, 'abc')
    self.assertIn('The third argument "axis2" must be an integer', str(raises.exception))
    with self.assertRaises(ValueError) as raises:
        cfunc(np.arange(4), 1, 0)
    self.assertIn('np.swapaxes: Argument axis1 out of bounds', str(raises.exception))
    with self.assertRaises(ValueError) as raises:
        cfunc(np.arange(8).reshape(2, 4), 0, -3)
    self.assertIn('np.swapaxes: Argument axis2 out of bounds', str(raises.exception))