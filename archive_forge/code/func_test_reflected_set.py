from numba.tests.support import TestCase
from numba import njit, types
from numba.typed import List, Dict
import numpy as np
def test_reflected_set(self):

    @njit
    def foo(x):
        if x:
            return 10
        else:
            return 20
    z = {1}
    self.assertEqual(foo(z), foo.py_func(z))
    self.assertEqual(foo.py_func(z), 10)

    @njit
    def foo():
        y = {1, 2}
        if y:
            return 10
        else:
            return 20
    self.assertEqual(foo(), foo.py_func())
    self.assertEqual(foo.py_func(), 10)

    @njit
    def foo():
        y = {1, 2}
        y.pop()
        y.pop()
        assert len(y) == 0
        if y:
            return 10
        else:
            return 20
    self.assertEqual(foo(), foo.py_func())
    self.assertEqual(foo.py_func(), 20)