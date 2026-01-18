from __future__ import division, absolute_import
import sys
import unittest
import operator
from incremental import getVersionString, IncomparableVersions
from incremental import Version, _inf
from twisted.trial.unittest import TestCase
def test_strWithDev(self):
    """
        Calling C{str} on a version with a dev release includes the dev
        release.
        """
    self.assertEqual(str(Version('dummy', 1, 0, 0, dev=1)), '[dummy, version 1.0.0.dev1]')