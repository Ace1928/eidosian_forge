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
class StrictTodoMixin(ResultsTestMixin):
    """
    Tests for the I{expected failure} features of
    L{twisted.trial.unittest.TestCase} in which the exact failure which is
    expected is indicated.
    """

    def setUp(self):
        """
        Setup our test case
        """
        self.loadSuite(self.StrictTodo)

    def test_counting(self):
        """
        Assert there are seven test cases
        """
        self.assertCount(7)

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

    def test_expectedFailures(self):
        """
        Ensure that expected failures are handled properly.
        """
        self.suite(self.reporter)
        expectedReasons = ['todo1', 'todo2', 'todo5']
        reasonsGotten = [r.reason for t, e, r in self.reporter.expectedFailures]
        self.assertEqual(expectedReasons, reasonsGotten)

    def test_unexpectedSuccesses(self):
        """
        Ensure that unexpected successes are caught.
        """
        self.suite(self.reporter)
        expectedReasons = [([RuntimeError], 'todo7')]
        reasonsGotten = [(r.errors, r.reason) for t, r in self.reporter.unexpectedSuccesses]
        self.assertEqual(expectedReasons, reasonsGotten)