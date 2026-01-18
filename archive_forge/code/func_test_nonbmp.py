import unittest
from ctypes import *
from ctypes.test import need_symbol
@unittest.skipIf(sizeof(c_wchar) < 4, 'sizeof(wchar_t) is smaller than 4 bytes')
def test_nonbmp(self):
    u = chr(1114111)
    w = c_wchar(u)
    self.assertEqual(w.value, u)