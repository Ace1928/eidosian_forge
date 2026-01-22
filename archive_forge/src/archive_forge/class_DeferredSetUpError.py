from __future__ import annotations
from twisted.internet import defer, reactor, threads
from twisted.python.failure import Failure
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import unittest
from twisted.trial.util import suppress as SUPPRESS
class DeferredSetUpError(unittest.TestCase):
    testCalled = False

    def setUp(self):
        return defer.fail(RuntimeError('deliberate error'))

    def test_ok(self):
        DeferredSetUpError.testCalled = True