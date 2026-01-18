import numpy as np
import unittest
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase
def test_static_raise_in_objmode_fallback(self):
    """
        Test code based on user submitted issue at
        https://github.com/numba/numba/issues/2159
        """

    def test0(n):
        return n

    def test1(n):
        if n == 0:
            raise ValueError()
        return test0(n)
    compiled = jit(forceobj=True)(test1)
    self.assertEqual(test1(10), compiled(10))
    self._ensure_objmode(compiled)