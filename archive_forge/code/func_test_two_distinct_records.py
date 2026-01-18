import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def test_two_distinct_records(self):
    """
        Testing the use of two scalar records of differing type
        """
    nbval1 = self.nbsample1d.copy()[0]
    nbval2 = self.refsample1d2.copy()[0]
    expected = nbval1['a'] + nbval2['f']
    nbrecord1 = numpy_support.from_dtype(recordtype)
    nbrecord2 = numpy_support.from_dtype(recordtype2)
    cfunc = self.get_cfunc(get_two_records_distinct, (nbrecord1, nbrecord2))
    got = cfunc(nbval1, nbval2)
    self.assertEqual(expected, got)