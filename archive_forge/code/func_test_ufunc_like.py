import unittest
import pickle
import numpy as np
from numba import void, float32, float64, int32, int64, jit, guvectorize
from numba.np.ufunc import GUVectorize
from numba.tests.support import tag, TestCase
def test_ufunc_like(self):
    gufunc = GUVectorize(axpy, '(), (), () -> ()', target=self.target)
    gufunc.add('(intp, intp, intp, intp[:])')
    gufunc = gufunc.build_ufunc()
    x = np.arange(10, dtype=np.intp)
    out = gufunc(x, x, x)
    np.testing.assert_equal(out, x * x + x)