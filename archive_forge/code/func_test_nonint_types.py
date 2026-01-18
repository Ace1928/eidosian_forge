from ctypes import *
from ctypes.test import need_symbol
from test import support
import unittest
import os
import _ctypes_test
def test_nonint_types(self):
    result = self.fail_fields(('a', c_char_p, 1))
    self.assertEqual(result, (TypeError, 'bit fields not allowed for type c_char_p'))
    result = self.fail_fields(('a', c_void_p, 1))
    self.assertEqual(result, (TypeError, 'bit fields not allowed for type c_void_p'))
    if c_int != c_long:
        result = self.fail_fields(('a', POINTER(c_int), 1))
        self.assertEqual(result, (TypeError, 'bit fields not allowed for type LP_c_int'))
    result = self.fail_fields(('a', c_char, 1))
    self.assertEqual(result, (TypeError, 'bit fields not allowed for type c_char'))

    class Dummy(Structure):
        _fields_ = []
    result = self.fail_fields(('a', Dummy, 1))
    self.assertEqual(result, (TypeError, 'bit fields not allowed for type Dummy'))