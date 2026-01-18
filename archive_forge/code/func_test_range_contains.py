import unittest
import sys
import numpy
from numba import jit, njit
from numba.core import types, utils
from numba.tests.support import tag
from numba.cpython.rangeobj import length_of_iterator
def test_range_contains(self):
    pyfunc = range_contains
    arglist = [(0, 0, 1), (-1, 0, 1), (1, 0, -1), (0, -1, 1), (0, 1, -1), (-1, 1, 1), (-1, 4, 1), (-1, 4, 10), (5, -5, -2)]
    bool_vals = [True, False]
    int_vals = [-10, -6, -5, -4, -2, -1, 0, 1, 2, 4, 5, 6, 10]
    float_vals = [-1.1, -1.0, 0.0, 1.0, 1.1]
    complex_vals = [1 + 0j, 1 + 1j, 1.1 + 0j, 1.0 + 1.1j]
    vallist = bool_vals + int_vals + float_vals + complex_vals
    cfunc = njit(pyfunc)
    for arg in arglist:
        for val in vallist:
            self.assertEqual(cfunc(val, *arg), pyfunc(val, *arg))
    non_numeric_vals = [{'a': 1}, [1], 'abc', (1,)]
    cfunc_obj = jit(pyfunc, forceobj=True)
    for arg in arglist:
        for val in non_numeric_vals:
            self.assertEqual(cfunc_obj(val, *arg), pyfunc(val, *arg))