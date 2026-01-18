import numpy as np
from numba.np.ufunc.ufuncbuilder import GUFuncBuilder
from numba import vectorize, guvectorize
from numba.np.ufunc import PyUFunc_One
from numba.np.ufunc.dufunc import DUFunc as UFuncBuilder
from numba.tests.support import tag, TestCase
from numba.core import config
import unittest
def test_ufunc_forceobj(self):
    from numba.tests.npyufunc.ufuncbuilding_usecases import add
    ufb = UFuncBuilder(add, targetoptions={'forceobj': True})
    cres = ufb.add('int32(int32, int32)')
    self.assertTrue(cres.objectmode)
    ufunc = ufb.build_ufunc()
    a = np.arange(10, dtype='int32')
    b = ufunc(a, a)
    self.assertPreciseEqual(a + a, b)