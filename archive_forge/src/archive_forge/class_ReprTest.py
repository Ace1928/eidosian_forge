from ctypes import *
import unittest
class ReprTest(unittest.TestCase):

    def test_numbers(self):
        for typ in subclasses:
            base = typ.__bases__[0]
            self.assertTrue(repr(base(42)).startswith(base.__name__))
            self.assertEqual('<X object at', repr(typ(42))[:12])

    def test_char(self):
        self.assertEqual("c_char(b'x')", repr(c_char(b'x')))
        self.assertEqual('<X object at', repr(X(b'x'))[:12])