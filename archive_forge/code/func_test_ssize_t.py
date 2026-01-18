from ctypes import *
import unittest
def test_ssize_t(self):
    self.assertEqual(sizeof(c_void_p), sizeof(c_ssize_t))