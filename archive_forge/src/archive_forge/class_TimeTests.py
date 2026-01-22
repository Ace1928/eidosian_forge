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
class TimeTests(TestCase):
    """
    Tests for the IReactorTime part of the reactor.
    """

    def test_seconds(self):
        """
        L{twisted.internet.reactor.seconds} should return something
        like a number.

        1. This test specifically does not assert any relation to the
           "system time" as returned by L{time.time} or
           L{twisted.python.runtime.seconds}, because at some point we
           may find a better option for scheduling calls than
           wallclock-time.
        2. This test *also* does not assert anything about the type of
           the result, because operations may not return ints or
           floats: For example, datetime-datetime == timedelta(0).
        """
        now = reactor.seconds()
        self.assertEqual(now - now + now, now)

    def test_callLaterUsesReactorSecondsInDelayedCall(self):
        """
        L{reactor.callLater<twisted.internet.interfaces.IReactorTime.callLater>}
        should use the reactor's seconds factory
        to produce the time at which the DelayedCall will be called.
        """
        oseconds = reactor.seconds
        reactor.seconds = lambda: 100
        try:
            call = reactor.callLater(5, lambda: None)
            self.assertEqual(call.getTime(), 105)
        finally:
            reactor.seconds = oseconds
        call.cancel()

    def test_callLaterUsesReactorSecondsAsDelayedCallSecondsFactory(self):
        """
        L{reactor.callLater<twisted.internet.interfaces.IReactorTime.callLater>}
        should propagate its own seconds factory
        to the DelayedCall to use as its own seconds factory.
        """
        oseconds = reactor.seconds
        reactor.seconds = lambda: 100
        try:
            call = reactor.callLater(5, lambda: None)
            self.assertEqual(call.seconds(), 100)
        finally:
            reactor.seconds = oseconds
        call.cancel()

    def test_callLater(self):
        """
        Test that a DelayedCall really calls the function it is
        supposed to call.
        """
        d = Deferred()
        reactor.callLater(0, d.callback, None)
        d.addCallback(self.assertEqual, None)
        return d

    def test_callLaterReset(self):
        """
        A L{DelayedCall} that is reset will be scheduled at the new time.
        """
        call = reactor.callLater(2, passthru, passthru)
        self.addCleanup(call.cancel)
        origTime = call.time
        call.reset(1)
        self.assertNotEqual(call.time, origTime)

    def test_cancelDelayedCall(self):
        """
        Test that when a DelayedCall is cancelled it does not run.
        """
        called = []

        def function():
            called.append(None)
        call = reactor.callLater(0, function)
        call.cancel()
        d = Deferred()

        def check():
            try:
                self.assertEqual(called, [])
            except BaseException:
                d.errback()
            else:
                d.callback(None)
        reactor.callLater(0, reactor.callLater, 0, check)
        return d

    def test_cancelCancelledDelayedCall(self):
        """
        Test that cancelling a DelayedCall which has already been cancelled
        raises the appropriate exception.
        """
        call = reactor.callLater(0, lambda: None)
        call.cancel()
        self.assertRaises(error.AlreadyCancelled, call.cancel)

    def test_cancelCalledDelayedCallSynchronous(self):
        """
        Test that cancelling a DelayedCall in the DelayedCall's function as
        that function is being invoked by the DelayedCall raises the
        appropriate exception.
        """
        d = Deferred()

        def later():
            try:
                self.assertRaises(error.AlreadyCalled, call.cancel)
            except BaseException:
                d.errback()
            else:
                d.callback(None)
        call = reactor.callLater(0, later)
        return d

    def test_cancelCalledDelayedCallAsynchronous(self):
        """
        Test that cancelling a DelayedCall after it has run its function
        raises the appropriate exception.
        """
        d = Deferred()

        def check():
            try:
                self.assertRaises(error.AlreadyCalled, call.cancel)
            except BaseException:
                d.errback()
            else:
                d.callback(None)

        def later():
            reactor.callLater(0, check)
        call = reactor.callLater(0, later)
        return d

    def testCallLaterTime(self):
        d = reactor.callLater(10, lambda: None)
        try:
            self.assertTrue(d.getTime() - (time.time() + 10) < 1)
        finally:
            d.cancel()

    def testDelayedCallStringification(self):
        dc = reactor.callLater(0, lambda x, y: None, 'x', y=10)
        str(dc)
        dc.reset(5)
        str(dc)
        dc.cancel()
        str(dc)
        dc = reactor.callLater(0, lambda: None, *range(10), x=[({'hello': 'world'}, 10j), reactor])
        str(dc)
        dc.cancel()
        str(dc)

        def calledBack(ignored):
            str(dc)
        d = Deferred().addCallback(calledBack)
        dc = reactor.callLater(0, d.callback, None)
        str(dc)
        return d

    def testDelayedCallSecondsOverride(self):
        """
        Test that the C{seconds} argument to DelayedCall gets used instead of
        the default timing function, if it is not None.
        """

        def seconds():
            return 10
        dc = base.DelayedCall(5, lambda: None, (), {}, lambda dc: None, lambda dc: None, seconds)
        self.assertEqual(dc.getTime(), 5)
        dc.reset(3)
        self.assertEqual(dc.getTime(), 13)