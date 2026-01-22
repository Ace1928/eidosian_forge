import pickle
from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.application import internet
from twisted.application.internet import (
from twisted.internet import task
from twisted.internet.defer import CancelledError, Deferred
from twisted.internet.interfaces import (
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransport
from twisted.logger import formatEvent, globalLogPublisher
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase, TestCase
class TimerServiceTests(TestCase):
    """
    Tests for L{twisted.application.internet.TimerService}.

    @type timer: L{TimerService}
    @ivar timer: service to test

    @type clock: L{task.Clock}
    @ivar clock: source of time

    @type deferred: L{Deferred}
    @ivar deferred: deferred returned by L{TimerServiceTests.call}.
    """

    def setUp(self):
        """
        Set up a timer service to test.
        """
        self.timer = TimerService(2, self.call)
        self.clock = self.timer.clock = task.Clock()
        self.deferred = Deferred()

    def call(self):
        """
        Function called by L{TimerService} being tested.

        @returns: C{self.deferred}
        @rtype: L{Deferred}
        """
        return self.deferred

    def test_startService(self):
        """
        When L{TimerService.startService} is called, it marks itself
        as running, creates a L{task.LoopingCall} and starts it.
        """
        self.timer.startService()
        self.assertTrue(self.timer.running, 'Service is started')
        self.assertIsInstance(self.timer._loop, task.LoopingCall)
        self.assertIdentical(self.clock, self.timer._loop.clock)
        self.assertTrue(self.timer._loop.running, 'LoopingCall is started')

    def test_startServiceRunsCallImmediately(self):
        """
        When L{TimerService.startService} is called, it calls the function
        immediately.
        """
        result = []
        self.timer.call = (result.append, (None,), {})
        self.timer.startService()
        self.assertEqual([None], result)

    def test_startServiceUsesGlobalReactor(self):
        """
        L{TimerService.startService} uses L{internet._maybeGlobalReactor} to
        choose the reactor to pass to L{task.LoopingCall}
        uses the global reactor.
        """
        otherClock = task.Clock()

        def getOtherClock(maybeReactor):
            return otherClock
        self.patch(internet, '_maybeGlobalReactor', getOtherClock)
        self.timer.startService()
        self.assertIdentical(otherClock, self.timer._loop.clock)

    def test_stopServiceWaits(self):
        """
        When L{TimerService.stopService} is called while a call is in progress.
        the L{Deferred} returned doesn't fire until after the call finishes.
        """
        self.timer.startService()
        d = self.timer.stopService()
        self.assertNoResult(d)
        self.assertEqual(True, self.timer.running)
        self.deferred.callback(object())
        self.assertIdentical(self.successResultOf(d), None)

    def test_stopServiceImmediately(self):
        """
        When L{TimerService.stopService} is called while a call isn't in progress.
        the L{Deferred} returned has already been fired.
        """
        self.timer.startService()
        self.deferred.callback(object())
        d = self.timer.stopService()
        self.assertIdentical(self.successResultOf(d), None)

    def test_failedCallLogsError(self):
        """
        When function passed to L{TimerService} returns a deferred that errbacks,
        the exception is logged, and L{TimerService.stopService} doesn't raise an error.
        """
        self.timer.startService()
        self.deferred.errback(Failure(ZeroDivisionError()))
        errors = self.flushLoggedErrors(ZeroDivisionError)
        self.assertEqual(1, len(errors))
        d = self.timer.stopService()
        self.assertIdentical(self.successResultOf(d), None)

    def test_pickleTimerServiceNotPickleLoop(self):
        """
        When pickling L{internet.TimerService}, it won't pickle
        L{internet.TimerService._loop}.
        """
        timer = TimerService(1, fakeTargetFunction)
        timer.startService()
        dumpedTimer = pickle.dumps(timer)
        timer.stopService()
        loadedTimer = pickle.loads(dumpedTimer)
        nothing = object()
        value = getattr(loadedTimer, '_loop', nothing)
        self.assertIdentical(nothing, value)

    def test_pickleTimerServiceNotPickleLoopFinished(self):
        """
        When pickling L{internet.TimerService}, it won't pickle
        L{internet.TimerService._loopFinished}.
        """
        timer = TimerService(1, fakeTargetFunction)
        timer.startService()
        dumpedTimer = pickle.dumps(timer)
        timer.stopService()
        loadedTimer = pickle.loads(dumpedTimer)
        nothing = object()
        value = getattr(loadedTimer, '_loopFinished', nothing)
        self.assertIdentical(nothing, value)