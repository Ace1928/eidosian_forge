import unittest
import sys
import numpy
from numba import jit, njit
from numba.core import types, utils
from numba.tests.support import tag
from numba.cpython.rangeobj import length_of_iterator
def test_range_attrs(self):
    pyfunc = range_attrs
    arglist = [(0, 0, 1), (0, -1, 1), (-1, 1, 1), (-1, 4, 1), (-1, 4, 10), (5, -5, -2)]
    cfunc = njit((types.int64, types.int64, types.int64))(pyfunc)
    for arg in arglist:
        self.assertEqual(cfunc(*arg), pyfunc(*arg))