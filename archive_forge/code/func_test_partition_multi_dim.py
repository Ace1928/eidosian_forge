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
def test_partition_multi_dim(self):
    pyfunc = partition
    cfunc = jit(nopython=True)(pyfunc)

    def check(a, kth):
        expected = pyfunc(a, kth)
        got = cfunc(a, kth)
        self.assertPreciseEqual(expected[:, :, kth], got[:, :, kth])
        for s in np.ndindex(expected.shape[:-1]):
            self.assertPreciseEqual(np.unique(expected[s][:kth]), np.unique(got[s][:kth]))
            self.assertPreciseEqual(np.unique(expected[s][kth:]), np.unique(got[s][kth:]))

    def a_variations(a):
        yield a
        yield a.T
        yield np.asfortranarray(a)
        yield np.full_like(a, fill_value=np.nan)
        yield np.full_like(a, fill_value=np.inf)
        yield (((1.0, 3.142, -np.inf, 3),),)
    a = np.linspace(1, 10, 48)
    a[4:7] = np.nan
    a[8] = -np.inf
    a[9] = np.inf
    a = a.reshape((4, 3, 4))
    for arr in a_variations(a):
        for k in range(-3, 3):
            check(arr, k)