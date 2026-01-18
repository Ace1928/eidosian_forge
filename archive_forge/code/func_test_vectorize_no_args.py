import numpy as np
from numba.np.ufunc.ufuncbuilder import GUFuncBuilder
from numba import vectorize, guvectorize
from numba.np.ufunc import PyUFunc_One
from numba.np.ufunc.dufunc import DUFunc as UFuncBuilder
from numba.tests.support import tag, TestCase
from numba.core import config
import unittest
def test_vectorize_no_args(self):
    from numba.tests.npyufunc.ufuncbuilding_usecases import add
    a = np.linspace(0, 1, 10)
    b = np.linspace(1, 2, 10)
    ufunc = vectorize(add)
    self.assertPreciseEqual(ufunc(a, b), a + b)
    ufunc2 = vectorize(add)
    c = np.empty(10)
    ufunc2(a, b, c)
    self.assertPreciseEqual(c, a + b)