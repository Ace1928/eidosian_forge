import gyp.common
import unittest
import sys
def test_platform_default(self):
    self.assertFlavor('freebsd', 'freebsd9', {})
    self.assertFlavor('freebsd', 'freebsd10', {})
    self.assertFlavor('openbsd', 'openbsd5', {})
    self.assertFlavor('solaris', 'sunos5', {})
    self.assertFlavor('solaris', 'sunos', {})
    self.assertFlavor('linux', 'linux2', {})
    self.assertFlavor('linux', 'linux3', {})
    self.assertFlavor('linux', 'linux', {})