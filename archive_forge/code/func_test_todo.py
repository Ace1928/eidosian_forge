from __future__ import annotations
import unittest as pyunit
from twisted.internet import defer
from twisted.python.failure import Failure
from twisted.trial import reporter, unittest, util
from twisted.trial.test import detests
def test_todo(self) -> None:
    result = self.runTest('test_expectedFailure')
    self.assertTrue(result.wasSuccessful())
    self.assertEqual(result.testsRun, 1)
    self.assertEqual(len(result.expectedFailures), 1)
    assert isinstance(result.expectedFailures[0][1], Failure)
    self._wasTimeout(result.expectedFailures[0][1])