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
class SuccessMixin:
    """
    Tests for the reporting of successful tests in L{twisted.trial.unittest.TestCase}.
    """

    def setUp(self):
        """
        Setup our test case
        """
        self.result = reporter.TestResult()

    def test_successful(self):
        """
        A successful test, used by other tests.
        """

    def assertSuccessful(self, test, result):
        """
        Utility function -- assert there is one success and the state is
        plausible
        """
        self.assertEqual(result.successes, 1)
        self.assertEqual(result.failures, [])
        self.assertEqual(result.errors, [])
        self.assertEqual(result.expectedFailures, [])
        self.assertEqual(result.unexpectedSuccesses, [])
        self.assertEqual(result.skips, [])

    def test_successfulIsReported(self):
        """
        Test that when a successful test is run, it is reported as a success,
        and not as any other kind of result.
        """
        test = self.__class__('test_successful')
        test.run(self.result)
        self.assertSuccessful(test, self.result)

    def test_defaultIsSuccessful(self):
        """
        The test case type can be instantiated with no arguments, run, and
        reported as being successful.
        """
        test = self.__class__()
        test.run(self.result)
        self.assertSuccessful(test, self.result)

    def test_noReference(self):
        """
        Test that no reference is kept on a successful test.
        """
        test = self.__class__('test_successful')
        ref = weakref.ref(test)
        test.run(self.result)
        self.assertSuccessful(test, self.result)
        del test
        gc.collect()
        self.assertIdentical(ref(), None)