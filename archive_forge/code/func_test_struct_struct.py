from ctypes import *
import unittest
def test_struct_struct(self):

    class POINT(Structure):
        _fields_ = [('x', c_int), ('y', c_int)]

    class RECT(Structure):
        _fields_ = [('ul', POINT), ('lr', POINT)]
    r = RECT()
    r.ul.x = 0
    r.ul.y = 1
    r.lr.x = 2
    r.lr.y = 3
    self.assertEqual(r._objects, None)
    r = RECT()
    pt = POINT(1, 2)
    r.ul = pt
    self.assertEqual(r._objects, {'0': {}})
    r.ul.x = 22
    r.ul.y = 44
    self.assertEqual(r._objects, {'0': {}})
    r.lr = POINT()
    self.assertEqual(r._objects, {'0': {}, '1': {}})