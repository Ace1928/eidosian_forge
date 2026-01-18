from __future__ import division, absolute_import
import sys
import unittest
import operator
from incremental import getVersionString, IncomparableVersions
from incremental import Version, _inf
from twisted.trial.unittest import TestCase
def test_rcOrpreButNotBoth(self):
    """
        Release Candidate and prerelease can't both be given.
        """
    with self.assertRaises(ValueError):
        Version('whatever', 1, 0, 0, prerelease=1, release_candidate=1)