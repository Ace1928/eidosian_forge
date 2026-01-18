import itertools
import pickle
import textwrap
import numpy as np
from numba import njit, vectorize
from numba.tests.support import MemoryLeakMixin, TestCase
from numba.core.errors import TypingError
import unittest
from numba.np.ufunc import dufunc
def test_reduce_dtype(self):
    duadd = vectorize('float64(float64, int64)', identity=0)(pyuadd)
    self._check_reduce(duadd, dtype=np.float64)