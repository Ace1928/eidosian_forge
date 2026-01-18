import sys
import unittest
import os
from distutils.errors import DistutilsPlatformError
from distutils.tests import support
def test_reg_class(self):
    from distutils.msvc9compiler import Reg
    self.assertRaises(KeyError, Reg.get_value, 'xxx', 'xxx')
    path = 'Control Panel\\Desktop'
    v = Reg.get_value(path, 'dragfullwindows')
    self.assertIn(v, ('0', '1', '2'))
    import winreg
    HKCU = winreg.HKEY_CURRENT_USER
    keys = Reg.read_keys(HKCU, 'xxxx')
    self.assertEqual(keys, None)
    keys = Reg.read_keys(HKCU, 'Control Panel')
    self.assertIn('Desktop', keys)