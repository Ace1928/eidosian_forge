from ctypes import *
import unittest
def test_size_t(self):
    self.assertEqual(sizeof(c_void_p), sizeof(c_size_t))