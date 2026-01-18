import ctypes
import struct
from numba.tests.support import TestCase
from numba import _helperlib
def test_simple_c_test(self):
    ret = self.numba_test_list()
    self.assertEqual(ret, 0)