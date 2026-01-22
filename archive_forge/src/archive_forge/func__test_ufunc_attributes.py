import numpy as np
from numba import float32, jit, njit
from numba.np.ufunc import Vectorize
from numba.core.errors import TypingError
from numba.tests.support import TestCase
import unittest
def _test_ufunc_attributes(self, cls, a, b, *args):
    """Test ufunc attributes"""
    vectorizer = cls(add, *args)
    vectorizer.add(float32(float32, float32))
    ufunc = vectorizer.build_ufunc()
    info = (cls, a.ndim)
    self.assertPreciseEqual(ufunc(a, b), a + b, msg=info)
    self.assertPreciseEqual(ufunc_reduce(ufunc, a), np.sum(a), msg=info)
    self.assertPreciseEqual(ufunc.accumulate(a), np.add.accumulate(a), msg=info)
    self.assertPreciseEqual(ufunc.outer(a, b), np.add.outer(a, b), msg=info)