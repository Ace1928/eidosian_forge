import unittest
from ctypes import *
from ctypes.test import need_symbol
import _ctypes_test
def test_setslice_cint(self):
    a = (c_int * 100)(*range(1100, 1200))
    b = list(range(1100, 1200))
    a[32:47] = list(range(32, 47))
    self.assertEqual(a[32:47], list(range(32, 47)))
    a[32:47] = range(132, 147)
    self.assertEqual(a[32:47], list(range(132, 147)))
    a[46:31:-1] = range(232, 247)
    self.assertEqual(a[32:47:1], list(range(246, 231, -1)))
    a[32:47] = range(1132, 1147)
    self.assertEqual(a[:], b)
    a[32:47:7] = range(3)
    b[32:47:7] = range(3)
    self.assertEqual(a[:], b)
    a[33::-3] = range(12)
    b[33::-3] = range(12)
    self.assertEqual(a[:], b)
    from operator import setitem
    self.assertRaises(TypeError, setitem, a, slice(0, 5), 'abcde')
    self.assertRaises(TypeError, setitem, a, slice(0, 5), ['a', 'b', 'c', 'd', 'e'])
    self.assertRaises(TypeError, setitem, a, slice(0, 5), [1, 2, 3, 4, 3.14])
    self.assertRaises(ValueError, setitem, a, slice(0, 5), range(32))