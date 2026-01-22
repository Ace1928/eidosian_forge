from __future__ import annotations
import unittest as pyunit
from twisted.internet import defer
from twisted.python.failure import Failure
from twisted.trial import reporter, unittest, util
from twisted.trial.test import detests
def _wasTimeout(self, error: Failure) -> None:
    self.assertEqual(error.check(defer.TimeoutError), defer.TimeoutError)