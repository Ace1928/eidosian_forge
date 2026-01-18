import numpy as np
import unittest
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase
def test_static_setitem_in_objmode_fallback(self):
    """
        Test code based on user submitted issue at
        https://github.com/numba/numba/issues/2169
        """

    def test0(n):
        return n

    def test(a1, a2):
        a1 = np.asarray(a1)
        a2[0] = 1
        return test0(a1.sum() + a2.sum())
    compiled = jit(forceobj=True)(test)
    args = (np.array([3]), np.array([4]))
    self.assertEqual(test(*args), compiled(*args))
    self._ensure_objmode(compiled)