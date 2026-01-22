from __future__ import annotations
from twisted.internet import defer, reactor, threads
from twisted.python.failure import Failure
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import unittest
from twisted.trial.util import suppress as SUPPRESS
class DeferredSetUpNeverFire(unittest.TestCase):
    testCalled = False

    def setUp(self):
        return defer.Deferred()

    def test_ok(self):
        DeferredSetUpNeverFire.testCalled = True