import os
import sys
import time
from unittest import skipIf
from twisted.internet import abstract, base, defer, error, interfaces, protocol, reactor
from twisted.internet.defer import Deferred, passthru
from twisted.internet.tcp import Connector
from twisted.python import util
from twisted.trial.unittest import TestCase
import %(reactor)s
from twisted.internet import reactor
class DelayedTests(TestCase):

    def setUp(self):
        self.finished = 0
        self.counter = 0
        self.timers = {}
        self.deferred = defer.Deferred()

    def tearDown(self):
        for t in self.timers.values():
            t.cancel()

    def checkTimers(self):
        l1 = self.timers.values()
        l2 = list(reactor.getDelayedCalls())
        missing = []
        for dc in l1:
            if dc not in l2:
                missing.append(dc)
        if missing:
            self.finished = 1
        self.assertFalse(missing, 'Should have been missing no calls, instead ' + 'was missing ' + repr(missing))

    def callback(self, tag):
        del self.timers[tag]
        self.checkTimers()

    def addCallback(self, tag):
        self.callback(tag)
        self.addTimer(15, self.callback)

    def done(self, tag):
        self.finished = 1
        self.callback(tag)
        self.deferred.callback(None)

    def addTimer(self, when, callback):
        self.timers[self.counter] = reactor.callLater(when * 0.01, callback, self.counter)
        self.counter += 1
        self.checkTimers()

    def testGetDelayedCalls(self):
        if not hasattr(reactor, 'getDelayedCalls'):
            return
        self.checkTimers()
        self.addTimer(35, self.done)
        self.addTimer(20, self.callback)
        self.addTimer(30, self.callback)
        which = self.counter
        self.addTimer(29, self.callback)
        self.addTimer(25, self.addCallback)
        self.addTimer(26, self.callback)
        self.timers[which].cancel()
        del self.timers[which]
        self.checkTimers()
        self.deferred.addCallback(lambda x: self.checkTimers())
        return self.deferred

    def test_active(self):
        """
        L{IDelayedCall.active} returns False once the call has run.
        """
        dcall = reactor.callLater(0.01, self.deferred.callback, True)
        self.assertTrue(dcall.active())

        def checkDeferredCall(success):
            self.assertFalse(dcall.active())
            return success
        self.deferred.addCallback(checkDeferredCall)
        return self.deferred