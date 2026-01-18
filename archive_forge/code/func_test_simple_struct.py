import unittest
from ctypes import *
from sys import getrefcount as grc
def test_simple_struct(self):

    class X(Structure):
        _fields_ = [('a', c_int), ('b', c_int)]
    a = 421234
    b = 421235
    x = X()
    self.assertEqual(x._objects, None)
    x.a = a
    x.b = b
    self.assertEqual(x._objects, None)