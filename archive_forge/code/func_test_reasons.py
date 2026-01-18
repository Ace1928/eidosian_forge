import gc
import sys
import unittest as pyunit
import weakref
from io import StringIO
from twisted.internet import defer, reactor
from twisted.python.compat import _PYPY
from twisted.python.reflect import namedAny
from twisted.trial import reporter, runner, unittest, util
from twisted.trial._asyncrunner import (
from twisted.trial.test import erroneous
from twisted.trial.test.test_suppression import SuppressionMixin
def test_reasons(self):
    """
        Test methods which raise L{unittest.SkipTest} or have their C{skip}
        attribute set to something are skipped.
        """
    self.suite(self.reporter)
    expectedReasons = ['class', 'skip2', 'class', 'class']
    reasonsGiven = [reason for test, reason in self.reporter.skips]
    self.assertEqual(expectedReasons, reasonsGiven)