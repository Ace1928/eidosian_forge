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
def test_np_trapz_x_dx_exceptions(self):
    pyfunc = np_trapz_x_dx
    cfunc = jit(nopython=True)(pyfunc)
    self.disable_leak_check()

    def check_not_ok(params):
        with self.assertRaises(ValueError) as e:
            cfunc(*params)
        self.assertIn('unable to broadcast', str(e.exception))
    y = [1, 2, 3, 4, 5]
    for x in ([4, 5, 6, 7, 8, 9], [4, 5, 6]):
        check_not_ok((y, x, 1.0))
    y = np.arange(60).reshape(3, 4, 5)
    x = np.arange(36).reshape(3, 4, 3)
    check_not_ok((y, x, 1.0))
    y = np.arange(60).reshape(3, 4, 5)
    x = np.array([4, 5, 6, 7])
    check_not_ok((y, x, 1.0))
    y = [1, 2, 3, 4, 5]
    dx = np.array([1.0, 2.0])
    check_not_ok((y, None, dx))
    y = np.arange(60).reshape(3, 4, 5)
    dx = np.arange(60).reshape(3, 4, 5)
    check_not_ok((y, None, dx))
    with self.assertTypingError() as e:
        y = np.array(4)
        check_not_ok((y, None, 1.0))
    self.assertIn('y cannot be 0D', str(e.exception))
    for y in (5, False, np.nan):
        with self.assertTypingError() as e:
            cfunc(y, None, 1.0)
        self.assertIn('y cannot be a scalar', str(e.exception))