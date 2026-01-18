from ctypes import *
import unittest
def test_zerosized_array(self):
    array = (c_int * 0)()
    self.assertRaises(IndexError, array.__setitem__, 0, None)
    self.assertRaises(IndexError, array.__getitem__, 0)
    self.assertRaises(IndexError, array.__setitem__, 1, None)
    self.assertRaises(IndexError, array.__getitem__, 1)
    self.assertRaises(IndexError, array.__setitem__, -1, None)
    self.assertRaises(IndexError, array.__getitem__, -1)