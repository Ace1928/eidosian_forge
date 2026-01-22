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
class ClassTodoMixin(ResultsTestMixin):
    """
    Tests for the class-wide I{expected failure} features of
    L{twisted.trial.unittest.TestCase}.
    """

    def setUp(self):
        """
        Setup our test case
        """
        self.loadSuite(self.TodoClass)

    def test_counting(self):
        """
        Ensure that we've got four test cases.
        """
        self.assertCount(4)

    def test_results(self):
        """
        Running a suite in which an entire class is marked as expected to fail
        produces a successful result with no recorded errors, failures, or
        skips, all methods which fail and were expected to fail recorded as
        C{expectedFailures}, and all methods which pass but which were expected
        to fail recorded as C{unexpectedSuccesses}.  Additionally, no tests are
        recorded as successes.
        """
        self.suite(self.reporter)
        self.assertTrue(self.reporter.wasSuccessful())
        self.assertEqual(self.reporter.errors, [])
        self.assertEqual(self.reporter.failures, [])
        self.assertEqual(self.reporter.skips, [])
        self.assertEqual(len(self.reporter.expectedFailures), 2)
        self.assertEqual(len(self.reporter.unexpectedSuccesses), 2)
        self.assertEqual(self.reporter.successes, 0)

    def test_expectedFailures(self):
        """
        Ensure that expected failures are handled properly.
        """
        self.suite(self.reporter)
        expectedReasons = ['method', 'class']
        reasonsGiven = [r.reason for t, e, r in self.reporter.expectedFailures]
        self.assertEqual(expectedReasons, reasonsGiven)

    def test_unexpectedSuccesses(self):
        """
        Ensure that unexpected successes are caught.
        """
        self.suite(self.reporter)
        expectedReasons = ['method', 'class']
        reasonsGiven = [r.reason for t, r in self.reporter.unexpectedSuccesses]
        self.assertEqual(expectedReasons, reasonsGiven)