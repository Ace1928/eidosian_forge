import unittest
import unittest.mock
import os.path
import sys
import test.support
from test.support import os_helper
from ctypes import *
from ctypes.util import find_library
def test_shell_injection(self):
    result = find_library('; echo Hello shell > ' + os_helper.TESTFN)
    self.assertFalse(os.path.lexists(os_helper.TESTFN))
    self.assertIsNone(result)