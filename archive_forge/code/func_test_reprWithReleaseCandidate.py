from __future__ import division, absolute_import
import sys
import unittest
import operator
from incremental import getVersionString, IncomparableVersions
from incremental import Version, _inf
from twisted.trial.unittest import TestCase
def test_reprWithReleaseCandidate(self):
    """
        Calling C{repr} on a version with a release candidate returns a
        human-readable string representation of the version including the rc.
        """
    self.assertEqual(repr(Version('dummy', 1, 2, 3, release_candidate=4)), "Version('dummy', 1, 2, 3, release_candidate=4)")