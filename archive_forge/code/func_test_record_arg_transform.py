import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def test_record_arg_transform(self):
    rec = numpy_support.from_dtype(recordtype3)
    transformed = mangle_type(rec)
    self.assertNotIn('first', transformed)
    self.assertNotIn('second', transformed)
    self.assertLess(len(transformed), 20)
    struct_arr = types.Array(rec, 1, 'C')
    transformed = mangle_type(struct_arr)
    self.assertIn('Array', transformed)
    self.assertNotIn('first', transformed)
    self.assertNotIn('second', transformed)
    self.assertLess(len(transformed), 50)