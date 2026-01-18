import sys
from itertools import product
import numpy as np
import unittest
from numba.core import types
from numba.core.errors import NumbaNotImplementedError
from numba.tests.support import TestCase
from numba.tests.enum_usecases import Shake, RequestError
from numba.np import numpy_support
def test_number_types(self):
    """
        Test from_dtype() and as_dtype() with the various scalar number types.
        """
    f = numpy_support.from_dtype

    def check(typechar, numba_type):
        dtype = np.dtype(typechar)
        self.assertIs(f(dtype), numba_type)
        self.assertIs(f(np.dtype('=' + typechar)), numba_type)
        self.assertEqual(dtype, numpy_support.as_dtype(numba_type))
    check('?', types.bool_)
    check('f', types.float32)
    check('f4', types.float32)
    check('d', types.float64)
    check('f8', types.float64)
    check('F', types.complex64)
    check('c8', types.complex64)
    check('D', types.complex128)
    check('c16', types.complex128)
    check('O', types.pyobject)
    check('b', types.int8)
    check('i1', types.int8)
    check('B', types.uint8)
    check('u1', types.uint8)
    check('h', types.int16)
    check('i2', types.int16)
    check('H', types.uint16)
    check('u2', types.uint16)
    check('i', types.int32)
    check('i4', types.int32)
    check('I', types.uint32)
    check('u4', types.uint32)
    check('q', types.int64)
    check('Q', types.uint64)
    for name in ('int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'intp', 'uintp'):
        self.assertIs(f(np.dtype(name)), getattr(types, name))
    foreign_align = '>' if sys.byteorder == 'little' else '<'
    for letter in 'hHiIlLqQfdFD':
        self.assertRaises(NumbaNotImplementedError, f, np.dtype(foreign_align + letter))