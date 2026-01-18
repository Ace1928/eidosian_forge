import itertools
import pickle
import textwrap
import numpy as np
from numba import njit, vectorize
from numba.tests.support import MemoryLeakMixin, TestCase
from numba.core.errors import TypingError
import unittest
from numba.np.ufunc import dufunc
def test_restricted(self):

    @vectorize(['float64(float64)'])
    def ident(x1):
        return x1
    self.check(ident, result_type=float)