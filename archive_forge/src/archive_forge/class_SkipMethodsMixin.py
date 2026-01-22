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
class SkipMethodsMixin(ResultsTestMixin):
    """
    Tests for the reporting of skipping tests in L{twisted.trial.unittest.TestCase}.
    """

    def setUp(self):
        """
        Setup our test case
        """
        self.loadSuite(self.Skipping)

    def test_counting(self):
        """
        Assert that there are three tests.
        """
        self.assertCount(3)

    def test_results(self):
        """
        Running a suite in which all methods are individually set to skip
        produces a successful result with no recorded errors or failures, all
        the skipped methods recorded as skips, and no methods recorded as
        successes.
        """
        self.suite(self.reporter)
        self.assertTrue(self.reporter.wasSuccessful())
        self.assertEqual(self.reporter.errors, [])
        self.assertEqual(self.reporter.failures, [])
        self.assertEqual(len(self.reporter.skips), 3)
        self.assertEqual(self.reporter.successes, 0)

    def test_setUp(self):
        """
        Running a suite in which all methods are skipped by C{setUp} raising
        L{SkipTest} produces a successful result with no recorded errors or
        failures, all skipped methods recorded as skips, and no methods recorded
        as successes.
        """
        self.loadSuite(self.SkippingSetUp)
        self.suite(self.reporter)
        self.assertTrue(self.reporter.wasSuccessful())
        self.assertEqual(self.reporter.errors, [])
        self.assertEqual(self.reporter.failures, [])
        self.assertEqual(len(self.reporter.skips), 2)
        self.assertEqual(self.reporter.successes, 0)

    def test_reasons(self):
        """
        Test that reasons work
        """
        self.suite(self.reporter)
        prefix = 'test_'
        for test, reason in self.reporter.skips:
            self.assertEqual(test.shortDescription()[len(prefix):], str(reason))

    def test_deprecatedSkipWithoutReason(self):
        """
        If a test method raises L{SkipTest} with no reason, a deprecation
        warning is emitted.
        """
        self.loadSuite(self.DeprecatedReasonlessSkip)
        self.suite(self.reporter)
        warnings = self.flushWarnings([self.DeprecatedReasonlessSkip.test_1])
        self.assertEqual(1, len(warnings))
        self.assertEqual(DeprecationWarning, warnings[0]['category'])
        self.assertEqual('Do not raise unittest.SkipTest with no arguments! Give a reason for skipping tests!', warnings[0]['message'])