import unittest
from distutils.version import LooseVersion
from distutils.version import StrictVersion
def test_prerelease(self):
    version = StrictVersion('1.2.3a1')
    self.assertEqual(version.version, (1, 2, 3))
    self.assertEqual(version.prerelease, ('a', 1))
    self.assertEqual(str(version), '1.2.3a1')
    version = StrictVersion('1.2.0')
    self.assertEqual(str(version), '1.2')