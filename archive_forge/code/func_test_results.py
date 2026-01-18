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
def test_results(self):
    """
        A test method which is marked as expected to fail with a particular
        exception is only counted as an expected failure if it does fail with
        that exception, not if it fails with some other exception.
        """
    self.suite(self.reporter)
    self.assertFalse(self.reporter.wasSuccessful())
    self.assertEqual(len(self.reporter.errors), 2)
    self.assertEqual(len(self.reporter.failures), 1)
    self.assertEqual(len(self.reporter.expectedFailures), 3)
    self.assertEqual(len(self.reporter.unexpectedSuccesses), 1)
    self.assertEqual(self.reporter.successes, 0)
    self.assertEqual(self.reporter.skips, [])