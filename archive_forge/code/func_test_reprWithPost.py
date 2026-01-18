from __future__ import division, absolute_import
import sys
import unittest
import operator
from incremental import getVersionString, IncomparableVersions
from incremental import Version, _inf
from twisted.trial.unittest import TestCase
def test_reprWithPost(self):
    """
        Calling C{repr} on a version with a postrelease returns a
        human-readable string representation of the version including the
        postrelease.
        """
    self.assertEqual(repr(Version('dummy', 1, 2, 3, post=4)), "Version('dummy', 1, 2, 3, post=4)")