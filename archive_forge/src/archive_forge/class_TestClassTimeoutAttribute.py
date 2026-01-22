from __future__ import annotations
from twisted.internet import defer, reactor, threads
from twisted.python.failure import Failure
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import unittest
from twisted.trial.util import suppress as SUPPRESS
class TestClassTimeoutAttribute(unittest.TestCase):
    timeout = 0.2

    def setUp(self):
        self.d = defer.Deferred()

    def testMethod(self):
        self.methodCalled = True
        return self.d