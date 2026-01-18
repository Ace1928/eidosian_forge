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
def test_tearDownRunsOnTestFailure(self):
    """
        L{SynchronousTestCase.tearDown} runs when a test method fails.
        """
    suite = self.loader.loadTestsFromTestCase(self.TestFailureButTearDownRuns)
    case = list(suite)[0]
    self.assertFalse(case.tornDown)
    suite.run(self.reporter)
    errors = self.reporter.errors
    self.assertTrue(len(errors) > 0)
    self.assertIsInstance(errors[0][1].value, erroneous.FoolishError)
    self.assertEqual(0, self.reporter.successes)
    self.assertTrue(case.tornDown)