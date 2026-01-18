import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def test_two_distinct_arrays(self):
    """
        Test with two arrays of distinct record types
        """
    pyfunc = get_two_arrays_distinct
    rec1 = numpy_support.from_dtype(recordtype)
    rec2 = numpy_support.from_dtype(recordtype2)
    cfunc = self.get_cfunc(pyfunc, (rec1[:], rec2[:], types.intp))
    for i in range(self.refsample1d.size):
        pres = pyfunc(self.refsample1d, self.refsample1d2, i)
        cres = cfunc(self.nbsample1d, self.nbsample1d2, i)
        self.assertEqual(pres, cres)