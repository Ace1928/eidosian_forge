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
def test_pop1(self):
    pyfunc = list_pop1
    cfunc = jit(nopython=True)(pyfunc)
    for n in [5, 40]:
        for i in [0, 1, n - 2, n - 1, -1, -2, -n + 3, -n + 1]:
            expected = pyfunc(n, i)
            self.assertPreciseEqual(cfunc(n, i), expected)