import sys
import operator
import pytest
import ctypes
import gc
import types
from typing import Any
import numpy as np
import numpy.dtypes
from numpy.core._rational_tests import rational
from numpy.core._multiarray_tests import create_custom_field_dtype
from numpy.testing import (
from numpy.compat import pickle
from itertools import permutations
import random
import hypothesis
from hypothesis.extra import numpy as hynp
class TestFromCTypes:

    @staticmethod
    def check(ctype, dtype):
        dtype = np.dtype(dtype)
        assert_equal(np.dtype(ctype), dtype)
        assert_equal(np.dtype(ctype()), dtype)

    def test_array(self):
        c8 = ctypes.c_uint8
        self.check(3 * c8, (np.uint8, (3,)))
        self.check(1 * c8, (np.uint8, (1,)))
        self.check(0 * c8, (np.uint8, (0,)))
        self.check(1 * (3 * c8), ((np.uint8, (3,)), (1,)))
        self.check(3 * (1 * c8), ((np.uint8, (1,)), (3,)))

    def test_padded_structure(self):

        class PaddedStruct(ctypes.Structure):
            _fields_ = [('a', ctypes.c_uint8), ('b', ctypes.c_uint16)]
        expected = np.dtype([('a', np.uint8), ('b', np.uint16)], align=True)
        self.check(PaddedStruct, expected)

    def test_bit_fields(self):

        class BitfieldStruct(ctypes.Structure):
            _fields_ = [('a', ctypes.c_uint8, 7), ('b', ctypes.c_uint8, 1)]
        assert_raises(TypeError, np.dtype, BitfieldStruct)
        assert_raises(TypeError, np.dtype, BitfieldStruct())

    def test_pointer(self):
        p_uint8 = ctypes.POINTER(ctypes.c_uint8)
        assert_raises(TypeError, np.dtype, p_uint8)

    def test_void_pointer(self):
        self.check(ctypes.c_void_p, np.uintp)

    def test_union(self):

        class Union(ctypes.Union):
            _fields_ = [('a', ctypes.c_uint8), ('b', ctypes.c_uint16)]
        expected = np.dtype(dict(names=['a', 'b'], formats=[np.uint8, np.uint16], offsets=[0, 0], itemsize=2))
        self.check(Union, expected)

    def test_union_with_struct_packed(self):

        class Struct(ctypes.Structure):
            _pack_ = 1
            _fields_ = [('one', ctypes.c_uint8), ('two', ctypes.c_uint32)]

        class Union(ctypes.Union):
            _fields_ = [('a', ctypes.c_uint8), ('b', ctypes.c_uint16), ('c', ctypes.c_uint32), ('d', Struct)]
        expected = np.dtype(dict(names=['a', 'b', 'c', 'd'], formats=['u1', np.uint16, np.uint32, [('one', 'u1'), ('two', np.uint32)]], offsets=[0, 0, 0, 0], itemsize=ctypes.sizeof(Union)))
        self.check(Union, expected)

    def test_union_packed(self):

        class Struct(ctypes.Structure):
            _fields_ = [('one', ctypes.c_uint8), ('two', ctypes.c_uint32)]
            _pack_ = 1

        class Union(ctypes.Union):
            _pack_ = 1
            _fields_ = [('a', ctypes.c_uint8), ('b', ctypes.c_uint16), ('c', ctypes.c_uint32), ('d', Struct)]
        expected = np.dtype(dict(names=['a', 'b', 'c', 'd'], formats=['u1', np.uint16, np.uint32, [('one', 'u1'), ('two', np.uint32)]], offsets=[0, 0, 0, 0], itemsize=ctypes.sizeof(Union)))
        self.check(Union, expected)

    def test_packed_structure(self):

        class PackedStructure(ctypes.Structure):
            _pack_ = 1
            _fields_ = [('a', ctypes.c_uint8), ('b', ctypes.c_uint16)]
        expected = np.dtype([('a', np.uint8), ('b', np.uint16)])
        self.check(PackedStructure, expected)

    def test_large_packed_structure(self):

        class PackedStructure(ctypes.Structure):
            _pack_ = 2
            _fields_ = [('a', ctypes.c_uint8), ('b', ctypes.c_uint16), ('c', ctypes.c_uint8), ('d', ctypes.c_uint16), ('e', ctypes.c_uint32), ('f', ctypes.c_uint32), ('g', ctypes.c_uint8)]
        expected = np.dtype(dict(formats=[np.uint8, np.uint16, np.uint8, np.uint16, np.uint32, np.uint32, np.uint8], offsets=[0, 2, 4, 6, 8, 12, 16], names=['a', 'b', 'c', 'd', 'e', 'f', 'g'], itemsize=18))
        self.check(PackedStructure, expected)

    def test_big_endian_structure_packed(self):

        class BigEndStruct(ctypes.BigEndianStructure):
            _fields_ = [('one', ctypes.c_uint8), ('two', ctypes.c_uint32)]
            _pack_ = 1
        expected = np.dtype([('one', 'u1'), ('two', '>u4')])
        self.check(BigEndStruct, expected)

    def test_little_endian_structure_packed(self):

        class LittleEndStruct(ctypes.LittleEndianStructure):
            _fields_ = [('one', ctypes.c_uint8), ('two', ctypes.c_uint32)]
            _pack_ = 1
        expected = np.dtype([('one', 'u1'), ('two', '<u4')])
        self.check(LittleEndStruct, expected)

    def test_little_endian_structure(self):

        class PaddedStruct(ctypes.LittleEndianStructure):
            _fields_ = [('a', ctypes.c_uint8), ('b', ctypes.c_uint16)]
        expected = np.dtype([('a', '<B'), ('b', '<H')], align=True)
        self.check(PaddedStruct, expected)

    def test_big_endian_structure(self):

        class PaddedStruct(ctypes.BigEndianStructure):
            _fields_ = [('a', ctypes.c_uint8), ('b', ctypes.c_uint16)]
        expected = np.dtype([('a', '>B'), ('b', '>H')], align=True)
        self.check(PaddedStruct, expected)

    def test_simple_endian_types(self):
        self.check(ctypes.c_uint16.__ctype_le__, np.dtype('<u2'))
        self.check(ctypes.c_uint16.__ctype_be__, np.dtype('>u2'))
        self.check(ctypes.c_uint8.__ctype_le__, np.dtype('u1'))
        self.check(ctypes.c_uint8.__ctype_be__, np.dtype('u1'))
    all_types = set(np.typecodes['All'])
    all_pairs = permutations(all_types, 2)

    @pytest.mark.parametrize('pair', all_pairs)
    def test_pairs(self, pair):
        """
        Check that np.dtype('x,y') matches [np.dtype('x'), np.dtype('y')]
        Example: np.dtype('d,I') -> dtype([('f0', '<f8'), ('f1', '<u4')])
        """
        pair_type = np.dtype('{},{}'.format(*pair))
        expected = np.dtype([('f0', pair[0]), ('f1', pair[1])])
        assert_equal(pair_type, expected)