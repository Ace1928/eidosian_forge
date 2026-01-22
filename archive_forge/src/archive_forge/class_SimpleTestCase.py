from ctypes import *
import unittest
class SimpleTestCase(unittest.TestCase):

    def test_cint(self):
        x = c_int()
        self.assertEqual(x._objects, None)
        x.value = 42
        self.assertEqual(x._objects, None)
        x = c_int(99)
        self.assertEqual(x._objects, None)

    def test_ccharp(self):
        x = c_char_p()
        self.assertEqual(x._objects, None)
        x.value = b'abc'
        self.assertEqual(x._objects, b'abc')
        x = c_char_p(b'spam')
        self.assertEqual(x._objects, b'spam')