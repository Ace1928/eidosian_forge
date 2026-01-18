from __future__ import division, absolute_import
import sys
import unittest
import operator
from incremental import getVersionString, IncomparableVersions
from incremental import Version, _inf
from twisted.trial.unittest import TestCase
def test_prereleaseAttributeDeprecated(self):
    """
        Accessing 'prerelease' on a Version is deprecated.
        """
    va = Version('whatever', 1, 0, 0, release_candidate=1)
    va.prerelease
    warnings = self.flushWarnings([self.test_prereleaseAttributeDeprecated])
    self.assertEqual(len(warnings), 1)
    self.assertEqual(warnings[0]['message'], 'Accessing incremental.Version.prerelease was deprecated in Incremental 16.9.0. Use Version.release_candidate instead.')