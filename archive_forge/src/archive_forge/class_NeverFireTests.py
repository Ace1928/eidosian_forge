from __future__ import annotations
import unittest as pyunit
from twisted.internet import defer
from twisted.python.failure import Failure
from twisted.trial import reporter, unittest, util
from twisted.trial.test import detests
class NeverFireTests(unittest.TestCase):

    def setUp(self) -> None:
        self._oldTimeout = util.DEFAULT_TIMEOUT_DURATION
        util.DEFAULT_TIMEOUT_DURATION = 0.1

    def tearDown(self) -> None:
        util.DEFAULT_TIMEOUT_DURATION = self._oldTimeout

    def _loadSuite(self, klass: type[pyunit.TestCase]) -> tuple[reporter.TestResult, pyunit.TestSuite]:
        loader = pyunit.TestLoader()
        r = reporter.TestResult()
        s = loader.loadTestsFromTestCase(klass)
        return (r, s)

    def test_setUp(self) -> None:
        self.assertFalse(detests.DeferredSetUpNeverFire.testCalled)
        result, suite = self._loadSuite(detests.DeferredSetUpNeverFire)
        suite(result)
        self.assertFalse(result.wasSuccessful())
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(len(result.failures), 0)
        self.assertEqual(len(result.errors), 1)
        self.assertFalse(detests.DeferredSetUpNeverFire.testCalled)
        assert isinstance(result.errors[0][1], Failure)
        self.assertTrue(result.errors[0][1].check(defer.TimeoutError))