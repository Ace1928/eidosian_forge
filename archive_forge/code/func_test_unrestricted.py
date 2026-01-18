import itertools
import pickle
import textwrap
import numpy as np
from numba import njit, vectorize
from numba.tests.support import MemoryLeakMixin, TestCase
from numba.core.errors import TypingError
import unittest
from numba.np.ufunc import dufunc
def test_unrestricted(self):

    @vectorize
    def ident(x1):
        return x1
    self.check(ident, result_type=(int, np.integer))