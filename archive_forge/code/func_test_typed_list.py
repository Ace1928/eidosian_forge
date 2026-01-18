from numba.tests.support import TestCase
from numba import njit, types
from numba.typed import List, Dict
import numpy as np
def test_typed_list(self):

    @njit
    def foo(x):
        if x:
            return 10
        else:
            return 20
    z = List.empty_list(types.int64)
    self.assertEqual(foo(z), foo.py_func(z))
    self.assertEqual(foo.py_func(z), 20)
    z.append(1)
    self.assertEqual(foo(z), foo.py_func(z))
    self.assertEqual(foo.py_func(z), 10)