import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def test_tuple_of_records(self):

    @njit
    def foo(rec_tup):
        x = 0
        for i in range(len(rec_tup)):
            x += rec_tup[i]['a']
        return x
    foo((self.a_rec1, self.a_rec2))
    foo.disable_compile()
    y = foo((self.ab_rec1, self.ab_rec2))
    self.assertEqual(2 * self.value + 1, y)