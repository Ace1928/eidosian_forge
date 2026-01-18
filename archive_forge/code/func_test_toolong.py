import unittest
from ctypes import *
from ctypes.test import need_symbol
@unittest.skip('test disabled')
def test_toolong(self):
    cs = c_wstring('abcdef')
    self.assertRaises(ValueError, setattr, cs, 'value', '123456789012345')
    self.assertRaises(ValueError, setattr, cs, 'value', '1234567')