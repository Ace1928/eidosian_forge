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
def test_sinc_exceptions(self):
    pyfunc = sinc
    cfunc = jit(nopython=True)(pyfunc)
    with self.assertRaises(TypingError) as raises:
        cfunc('str')
    self.assertIn('Argument "x" must be a Number or array-like', str(raises.exception))
    self.disable_leak_check()