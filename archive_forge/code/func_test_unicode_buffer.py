from ctypes import *
from ctypes.test import need_symbol
import unittest
@need_symbol('c_wchar')
def test_unicode_buffer(self):
    b = create_unicode_buffer(32)
    self.assertEqual(len(b), 32)
    self.assertEqual(sizeof(b), 32 * sizeof(c_wchar))
    self.assertIs(type(b[0]), str)
    b = create_unicode_buffer('abc')
    self.assertEqual(len(b), 4)
    self.assertEqual(sizeof(b), 4 * sizeof(c_wchar))
    self.assertIs(type(b[0]), str)
    self.assertEqual(b[0], 'a')
    self.assertEqual(b[:], 'abc\x00')
    self.assertEqual(b[:], 'abc\x00')
    self.assertEqual(b[::-1], '\x00cba')
    self.assertEqual(b[::2], 'ac')
    self.assertEqual(b[::5], 'a')
    self.assertRaises(TypeError, create_unicode_buffer, b'abc')