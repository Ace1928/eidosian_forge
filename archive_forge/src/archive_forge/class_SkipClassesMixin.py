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
class SkipClassesMixin(ResultsTestMixin):
    """
    Test the class skipping features of L{twisted.trial.unittest.TestCase}.
    """

    def setUp(self):
        """
        Setup our test case
        """
        self.loadSuite(self.SkippedClass)
        self.SkippedClass._setUpRan = False

    def test_counting(self):
        """
        Skipped test methods still contribute to the total test count.
        """
        self.assertCount(4)

    def test_setUpRan(self):
        """
        The C{setUp} method is not called if the class is set to skip.
        """
        self.suite(self.reporter)
        self.assertFalse(self.SkippedClass._setUpRan)

    def test_results(self):
        """
        Skipped test methods don't cause C{wasSuccessful} to return C{False},
        nor do they contribute to the C{errors} or C{failures} of the reporter,
        or to the count of successes.  They do, however, add elements to the
        reporter's C{skips} list.
        """
        self.suite(self.reporter)
        self.assertTrue(self.reporter.wasSuccessful())
        self.assertEqual(self.reporter.errors, [])
        self.assertEqual(self.reporter.failures, [])
        self.assertEqual(len(self.reporter.skips), 4)
        self.assertEqual(self.reporter.successes, 0)

    def test_reasons(self):
        """
        Test methods which raise L{unittest.SkipTest} or have their C{skip}
        attribute set to something are skipped.
        """
        self.suite(self.reporter)
        expectedReasons = ['class', 'skip2', 'class', 'class']
        reasonsGiven = [reason for test, reason in self.reporter.skips]
        self.assertEqual(expectedReasons, reasonsGiven)