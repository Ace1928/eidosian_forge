from __future__ import annotations
from twisted.internet import defer, reactor, threads
from twisted.python.failure import Failure
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import unittest
from twisted.trial.util import suppress as SUPPRESS
class DeferredSetUpSkip(unittest.TestCase):
    testCalled = False

    def setUp(self):
        d = defer.succeed('value')
        d.addCallback(self._cb1)
        return d

    def _cb1(self, ignored):
        raise unittest.SkipTest('skip me')

    def test_ok(self):
        DeferredSetUpSkip.testCalled = True