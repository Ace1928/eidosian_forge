from collections import namedtuple
import contextlib
import itertools
import math
import sys
import ctypes as ct
import numpy as np
from numba import jit, typeof, njit, literal_unroll, literally
import unittest
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.experimental import jitclass
from numba.core.extending import overload
def test_setslice2(self):
    pyfunc = list_setslice2
    cfunc = jit(nopython=True)(pyfunc)
    sizes = [5, 40]
    for n, n_src in itertools.product(sizes, sizes):
        indices = [0, 1, n - 2, -1, -2, -n + 3, -n - 1, -n]
        for start, stop in itertools.product(indices, indices):
            expected = pyfunc(n, n_src, start, stop)
            self.assertPreciseEqual(cfunc(n, n_src, start, stop), expected)