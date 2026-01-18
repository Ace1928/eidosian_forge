import itertools
import numpy as np
import unittest
from numba import jit, typeof, njit
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import MemoryLeakMixin, TestCase
def test_setitem_0d(self):
    pyfunc = setitem_usecase
    cfunc = jit(nopython=True)(pyfunc)
    inps = [(np.zeros(3), np.array(3.14)), (np.zeros(2), np.array(2)), (np.zeros(3, dtype=np.int64), np.array(3, dtype=np.int64)), (np.zeros(3, dtype=np.float64), np.array(1, dtype=np.int64)), (np.zeros(5, dtype='<U3'), np.array('abc')), (np.zeros((3,), dtype='<U3'), np.array('a')), (np.array(['abc', 'def', 'ghi'], dtype='<U3'), np.array('WXYZ', dtype='<U4')), (np.zeros(3, dtype=complex), np.array(2 + 3j, dtype=complex))]
    for x1, v in inps:
        x2 = x1.copy()
        pyfunc(x1, 0, v)
        cfunc(x2, 0, v)
        self.assertPreciseEqual(x1, x2)