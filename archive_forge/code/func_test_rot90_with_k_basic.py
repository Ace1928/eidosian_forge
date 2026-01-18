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
def test_rot90_with_k_basic(self):
    pyfunc = rot90_k
    cfunc = jit(nopython=True)(pyfunc)

    def a_variations():
        yield np.arange(10).reshape(5, 2)
        yield np.arange(20).reshape(5, 2, 2)
        yield np.arange(64).reshape(2, 2, 2, 2, 2, 2)
    for a in a_variations():
        for k in range(-5, 6):
            expected = pyfunc(a, k)
            got = cfunc(a, k)
            self.assertPreciseEqual(expected, got)