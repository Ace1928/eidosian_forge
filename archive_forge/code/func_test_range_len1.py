import unittest
import sys
import numpy
from numba import jit, njit
from numba.core import types, utils
from numba.tests.support import tag
from numba.cpython.rangeobj import length_of_iterator
def test_range_len1(self):
    pyfunc = range_len1
    typelist = [types.int16, types.int32, types.int64]
    arglist = [5, 0, -5]
    for typ in typelist:
        cfunc = njit((typ,))(pyfunc)
        for arg in arglist:
            self.assertEqual(cfunc(typ(arg)), pyfunc(typ(arg)))