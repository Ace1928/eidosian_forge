import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def test_record_dtype_with_titles_roundtrip(self):
    recdtype = np.dtype([(('title a', 'a'), np.float_), ('b', np.float_)])
    nbtype = numpy_support.from_dtype(recdtype)
    self.assertTrue(nbtype.is_title('title a'))
    self.assertFalse(nbtype.is_title('a'))
    self.assertFalse(nbtype.is_title('b'))
    got = numpy_support.as_dtype(nbtype)
    self.assertTrue(got, recdtype)