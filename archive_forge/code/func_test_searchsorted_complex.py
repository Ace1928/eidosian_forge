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
def test_searchsorted_complex(self):
    pyfunc = searchsorted
    cfunc = jit(nopython=True)(pyfunc)
    pyfunc_left = searchsorted_left
    cfunc_left = jit(nopython=True)(pyfunc_left)
    pyfunc_right = searchsorted_right
    cfunc_right = jit(nopython=True)(pyfunc_right)

    def check(a, v):
        expected = pyfunc(a, v)
        got = cfunc(a, v)
        self.assertPreciseEqual(expected, got)
        expected = pyfunc_left(a, v)
        got = cfunc_left(a, v)
        self.assertPreciseEqual(expected, got)
        expected = pyfunc_right(a, v)
        got = cfunc_right(a, v)
        self.assertPreciseEqual(expected, got)
    pool = [0, 1, np.nan]
    element_pool = [complex(*c) for c in itertools.product(pool, pool)]
    for i in range(100):
        sample_size = self.rnd.choice([3, 5, len(element_pool)])
        a = self.rnd.choice(element_pool, sample_size)
        v = self.rnd.choice(element_pool, sample_size + (i % 3 - 1))
        check(a, v)
        check(np.sort(a), v)
    check(a=np.array(element_pool), v=np.arange(2))