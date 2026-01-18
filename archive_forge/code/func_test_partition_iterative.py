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
def test_partition_iterative(self):
    pyfunc = partition
    cfunc = jit(nopython=True)(pyfunc)
    assert_partitioned = partial(self.assert_partitioned, pyfunc, cfunc)
    d = np.array([3, 4, 2, 1])
    p = cfunc(d, (0, 3))
    assert_partitioned(p, (0, 3))
    assert_partitioned(d[np.argpartition(d, (0, 3))], (0, 3))
    self.assertPreciseEqual(p, cfunc(d, (-3, -1)))
    d = np.arange(17)
    self.rnd.shuffle(d)
    self.assertPreciseEqual(np.arange(17), cfunc(d, list(range(d.size))))
    d = np.arange(17)
    self.rnd.shuffle(d)
    keys = np.array([1, 3, 8, -2])
    self.rnd.shuffle(d)
    p = cfunc(d, keys)
    assert_partitioned(p, keys)
    self.rnd.shuffle(keys)
    self.assertPreciseEqual(cfunc(d, keys), p)
    d = np.arange(20)[::-1]
    assert_partitioned(cfunc(d, [5] * 4), [5])
    assert_partitioned(cfunc(d, [5] * 4 + [6, 13]), [5] * 4 + [6, 13])