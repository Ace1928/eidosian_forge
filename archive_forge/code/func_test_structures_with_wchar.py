import platform
from platform import architecture as _architecture
import struct
import sys
import unittest
from ctypes.test import need_symbol
from ctypes import (CDLL, Array, Structure, Union, POINTER, sizeof, byref, alignment,
from ctypes.util import find_library
from struct import calcsize
import _ctypes_test
from collections import namedtuple
from test import support
@need_symbol('c_wchar')
def test_structures_with_wchar(self):

    class PersonW(Structure):
        _fields_ = [('name', c_wchar * 12), ('age', c_int)]
    p = PersonW('Someone é')
    self.assertEqual(p.name, 'Someone é')
    self.assertEqual(PersonW('1234567890').name, '1234567890')
    self.assertEqual(PersonW('12345678901').name, '12345678901')
    self.assertEqual(PersonW('123456789012').name, '123456789012')
    self.assertRaises(ValueError, PersonW, '1234567890123')