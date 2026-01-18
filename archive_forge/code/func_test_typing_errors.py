import contextlib
import sys
import numpy as np
import random
import re
import threading
import gc
from numba.core.errors import TypingError
from numba import njit
from numba.core import types, utils, config
from numba.tests.support import MemoryLeakMixin, TestCase, tag, skip_if_32bit
import unittest
def test_typing_errors(self):
    pyfunc = np_concatenate1
    cfunc = nrtjit(pyfunc)
    a = np.arange(15)
    b = a.reshape((3, 5))
    c = a.astype(np.dtype([('x', np.int8)]))
    d = np.array(42)
    with self.assertTypingError() as raises:
        cfunc(a, b, b)
    self.assertIn('all the input arrays must have same number of dimensions', str(raises.exception))
    with self.assertTypingError() as raises:
        cfunc(a, c, c)
    self.assertIn('input arrays must have compatible dtypes', str(raises.exception))
    with self.assertTypingError() as raises:
        cfunc(d, d, d)
    self.assertIn('zero-dimensional arrays cannot be concatenated', str(raises.exception))
    with self.assertTypingError() as raises:
        cfunc(c, 1, c)
    self.assertIn('expecting a non-empty tuple of arrays', str(raises.exception))