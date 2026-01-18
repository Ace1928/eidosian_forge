import numpy as np
from numba.np.ufunc.ufuncbuilder import GUFuncBuilder
from numba import vectorize, guvectorize
from numba.np.ufunc import PyUFunc_One
from numba.np.ufunc.dufunc import DUFunc as UFuncBuilder
from numba.tests.support import tag, TestCase
from numba.core import config
import unittest
def test_vectorize_output_kwarg(self):
    """
        Passing the output array as a keyword argument (issue #1867).
        """

    def check(ufunc):
        a = np.arange(10, 16, dtype='int32')
        out = np.zeros_like(a)
        got = ufunc(a, a, out=out)
        self.assertIs(got, out)
        self.assertPreciseEqual(out, a + a)
        with self.assertRaises(TypeError):
            ufunc(a, a, zzz=out)
    from numba.tests.npyufunc.ufuncbuilding_usecases import add
    ufunc = vectorize(['int32(int32, int32)'], nopython=True)(add)
    check(ufunc)
    ufunc = vectorize(nopython=True)(add)
    check(ufunc)
    check(ufunc)