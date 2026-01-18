from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_readonly_after_ravel(self):

    def unfold_ravel(x, y):
        r, c = x.shape
        a = np.broadcast_to(x, (y, r, c))
        b = np.swapaxes(a, 0, 1)
        cc = b.ravel()
        d = np.reshape(cc, (-1, c))
        d[y - 1:, :] = d[:1 - y]
        return d
    pyfunc = unfold_ravel
    cfunc = jit(nopython=True)(pyfunc)
    res_nb = cfunc(np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]), 2)
    res_py = pyfunc(np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]), 2)
    np.testing.assert_array_equal(res_py, res_nb)