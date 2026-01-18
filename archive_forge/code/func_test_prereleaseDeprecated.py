from __future__ import division, absolute_import
import sys
import unittest
import operator
from incremental import getVersionString, IncomparableVersions
from incremental import Version, _inf
from twisted.trial.unittest import TestCase
def test_prereleaseDeprecated(self):
    """
        Passing 'prerelease' to Version is deprecated.
        """
    Version('whatever', 1, 0, 0, prerelease=1)
    warnings = self.flushWarnings([self.test_prereleaseDeprecated])
    self.assertEqual(len(warnings), 1)
    self.assertEqual(warnings[0]['message'], 'Passing prerelease to incremental.Version was deprecated in Incremental 16.9.0. Please pass release_candidate instead.')