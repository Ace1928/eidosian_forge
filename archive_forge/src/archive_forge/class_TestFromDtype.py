import sys
from itertools import product
import numpy as np
import unittest
from numba.core import types
from numba.core.errors import NumbaNotImplementedError
from numba.tests.support import TestCase
from numba.tests.enum_usecases import Shake, RequestError
from numba.np import numpy_support
class TestFromDtype(TestCase):

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

    def test_string_types(self):
        """
        Test from_dtype() and as_dtype() with the character string types.
        """

        def check(typestring, numba_type):
            dtype = np.dtype(typestring)
            self.assertEqual(numpy_support.from_dtype(dtype), numba_type)
            self.assertEqual(dtype, numpy_support.as_dtype(numba_type))
        check('S10', types.CharSeq(10))
        check('a11', types.CharSeq(11))
        check('U12', types.UnicodeCharSeq(12))

    def check_datetime_types(self, letter, nb_class):

        def check(dtype, numba_type, code):
            tp = numpy_support.from_dtype(dtype)
            self.assertEqual(tp, numba_type)
            self.assertEqual(tp.unit_code, code)
            self.assertEqual(numpy_support.as_dtype(numba_type), dtype)
            self.assertEqual(numpy_support.as_dtype(tp), dtype)
        check(np.dtype(letter), nb_class(''), 14)

    def test_datetime_types(self):
        """
        Test from_dtype() and as_dtype() with the datetime types.
        """
        self.check_datetime_types('M', types.NPDatetime)

    def test_timedelta_types(self):
        """
        Test from_dtype() and as_dtype() with the timedelta types.
        """
        self.check_datetime_types('m', types.NPTimedelta)

    def test_struct_types(self):

        def check(dtype, fields, size, aligned):
            tp = numpy_support.from_dtype(dtype)
            self.assertIsInstance(tp, types.Record)
            self.assertEqual(tp.dtype, dtype)
            self.assertEqual(tp.fields, fields)
            self.assertEqual(tp.size, size)
            self.assertEqual(tp.aligned, aligned)
        dtype = np.dtype([('a', np.int16), ('b', np.int32)])
        check(dtype, fields={'a': (types.int16, 0, None, None), 'b': (types.int32, 2, None, None)}, size=6, aligned=False)
        dtype = np.dtype([('a', np.int16), ('b', np.int32)], align=True)
        check(dtype, fields={'a': (types.int16, 0, None, None), 'b': (types.int32, 4, None, None)}, size=8, aligned=True)
        dtype = np.dtype([('m', np.int32), ('n', 'S5')])
        check(dtype, fields={'m': (types.int32, 0, None, None), 'n': (types.CharSeq(5), 4, None, None)}, size=9, aligned=False)

    def test_enum_type(self):

        def check(base_inst, enum_def, type_class):
            np_dt = np.dtype(base_inst)
            nb_ty = numpy_support.from_dtype(np_dt)
            inst = type_class(enum_def, nb_ty)
            recovered = numpy_support.as_dtype(inst)
            self.assertEqual(np_dt, recovered)
        dts = [np.float64, np.int32, np.complex128, np.bool_]
        enums = [Shake, RequestError]
        for dt, enum in product(dts, enums):
            check(dt, enum, types.EnumMember)
        for dt, enum in product(dts, enums):
            check(dt, enum, types.IntEnumMember)