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
class FixtureMixin:
    """
    Tests for fixture helper methods (e.g. setUp, tearDown).
    """

    def setUp(self):
        """
        Setup our test case
        """
        self.reporter = reporter.Reporter()
        self.loader = pyunit.TestLoader()

    def test_brokenSetUp(self):
        """
        When setUp fails, the error is recorded in the result object.
        """
        suite = self.loader.loadTestsFromTestCase(self.TestFailureInSetUp)
        suite.run(self.reporter)
        self.assertTrue(len(self.reporter.errors) > 0)
        self.assertIsInstance(self.reporter.errors[0][1].value, erroneous.FoolishError)
        self.assertEqual(0, self.reporter.successes)

    def test_brokenTearDown(self):
        """
        When tearDown fails, the error is recorded in the result object.
        """
        suite = self.loader.loadTestsFromTestCase(self.TestFailureInTearDown)
        suite.run(self.reporter)
        errors = self.reporter.errors
        self.assertTrue(len(errors) > 0)
        self.assertIsInstance(errors[0][1].value, erroneous.FoolishError)
        self.assertEqual(0, self.reporter.successes)

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