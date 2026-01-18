from __future__ import division, absolute_import
import sys
import unittest
import operator
from incremental import getVersionString, IncomparableVersions
from incremental import Version, _inf
from twisted.trial.unittest import TestCase
@unittest.skipIf(sys.version_info < (3,), 'Comparisons do not raise on py2')
def test_versionComparisonNonVersion(self):
    """
        Versions can be compared with non-versions.
        """
    v = Version('dummy', 1, 0, 0)
    o = object()
    with self.assertRaises(TypeError):
        v > o
    with self.assertRaises(TypeError):
        v < o
    with self.assertRaises(TypeError):
        v >= o
    with self.assertRaises(TypeError):
        v <= o
    self.assertFalse(v == o)
    self.assertTrue(v != o)