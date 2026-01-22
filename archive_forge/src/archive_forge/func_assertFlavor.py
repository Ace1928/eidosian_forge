import gyp.common
import unittest
import sys
def assertFlavor(self, expected, argument, param):
    sys.platform = argument
    self.assertEqual(expected, gyp.common.GetFlavor(param))