from ctypes import *
from ctypes.test import need_symbol
from test import support
import unittest
import os
import _ctypes_test
def test_single_bitfield_size(self):
    for c_typ in int_types:
        result = self.fail_fields(('a', c_typ, -1))
        self.assertEqual(result, (ValueError, 'number of bits invalid for bit field'))
        result = self.fail_fields(('a', c_typ, 0))
        self.assertEqual(result, (ValueError, 'number of bits invalid for bit field'))

        class X(Structure):
            _fields_ = [('a', c_typ, 1)]
        self.assertEqual(sizeof(X), sizeof(c_typ))

        class X(Structure):
            _fields_ = [('a', c_typ, sizeof(c_typ) * 8)]
        self.assertEqual(sizeof(X), sizeof(c_typ))
        result = self.fail_fields(('a', c_typ, sizeof(c_typ) * 8 + 1))
        self.assertEqual(result, (ValueError, 'number of bits invalid for bit field'))