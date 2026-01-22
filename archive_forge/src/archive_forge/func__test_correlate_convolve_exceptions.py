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
def _test_correlate_convolve_exceptions(self, fn):
    self.disable_leak_check()
    _a = np.ones(shape=(0,))
    _b = np.arange(5)
    cfunc = jit(nopython=True)(fn)
    for x, y in [(_a, _b), (_b, _a)]:
        with self.assertRaises(ValueError) as raises:
            cfunc(x, y)
        if len(x) == 0:
            self.assertIn("'a' cannot be empty", str(raises.exception))
        else:
            self.assertIn("'v' cannot be empty", str(raises.exception))
    with self.assertRaises(ValueError) as raises:
        cfunc(_b, _b, mode='invalid mode')
        self.assertIn("Invalid 'mode'", str(raises.exception))