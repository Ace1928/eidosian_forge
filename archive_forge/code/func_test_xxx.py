import unittest
from ctypes import *
from sys import getrefcount as grc
def test_xxx(self):

    class X(Structure):
        _fields_ = [('a', c_char_p), ('b', c_char_p)]

    class Y(Structure):
        _fields_ = [('x', X), ('y', X)]
    s1 = b'Hello, World'
    s2 = b'Hallo, Welt'
    x = X()
    x.a = s1
    x.b = s2
    self.assertEqual(x._objects, {'0': s1, '1': s2})
    y = Y()
    y.x = x
    self.assertEqual(y._objects, {'0': {'0': s1, '1': s2}})