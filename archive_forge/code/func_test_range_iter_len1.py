import unittest
import sys
import numpy
from numba import jit, njit
from numba.core import types, utils
from numba.tests.support import tag
from numba.cpython.rangeobj import length_of_iterator
def test_range_iter_len1(self):
    range_func = range_len1
    range_iter_func = range_iter_len1
    typelist = [types.int16, types.int32, types.int64]
    arglist = [5, 0, -5]
    for typ in typelist:
        cfunc = njit((typ,))(range_iter_func)
        for arg in arglist:
            self.assertEqual(cfunc(typ(arg)), range_func(typ(arg)))