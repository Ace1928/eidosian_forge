from __future__ import division, absolute_import
import sys
import unittest
import operator
from incremental import getVersionString, IncomparableVersions
from incremental import Version, _inf
from twisted.trial.unittest import TestCase
def test_reprWithDev(self):
    """
        Calling C{repr} on a version with a dev release returns a
        human-readable string representation of the version including the dev
        release.
        """
    self.assertEqual(repr(Version('dummy', 1, 2, 3, dev=4)), "Version('dummy', 1, 2, 3, dev=4)")