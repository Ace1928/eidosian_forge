import numpy as np
from numba.np.ufunc.ufuncbuilder import GUFuncBuilder
from numba import vectorize, guvectorize
from numba.np.ufunc import PyUFunc_One
from numba.np.ufunc.dufunc import DUFunc as UFuncBuilder
from numba.tests.support import tag, TestCase
from numba.core import config
import unittest
def test_ufunc_struct(self):
    from numba.tests.npyufunc.ufuncbuilding_usecases import add
    ufb = UFuncBuilder(add)
    cres = ufb.add('complex64(complex64, complex64)')
    self.assertFalse(cres.objectmode)
    ufunc = ufb.build_ufunc()

    def check(a):
        b = ufunc(a, a)
        self.assertPreciseEqual(a + a, b)
        self.assertEqual(b.dtype, a.dtype)
    a = np.arange(12, dtype='complex64') + 1j
    check(a)
    a = a[::2]
    check(a)
    a = a.reshape((2, 3))
    check(a)