from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
class LoopTests(unittest.TestCase):
    """
    Tests for L{task.LoopingCall} based on a fake L{IReactorTime}
    implementation.
    """

    def test_defaultClock(self):
        """
        L{LoopingCall}'s default clock should be the reactor.
        """
        call = task.LoopingCall(lambda: None)
        self.assertEqual(call.clock, reactor)

    def test_callbackTimeSkips(self):
        """
        When more time than the defined interval passes during the execution
        of a callback, L{LoopingCall} should schedule the next call for the
        next interval which is still in the future.
        """
        times = []
        callDuration = None
        clock = task.Clock()

        def aCallback():
            times.append(clock.seconds())
            clock.advance(callDuration)
        call = task.LoopingCall(aCallback)
        call.clock = clock
        callDuration = 2
        call.start(0.5)
        self.assertEqual(times, [0])
        self.assertEqual(clock.seconds(), callDuration)
        clock.advance(0)
        self.assertEqual(times, [0])
        callDuration = 1
        clock.advance(0.5)
        self.assertEqual(times, [0, 2.5])
        self.assertEqual(clock.seconds(), 3.5)
        clock.advance(0)
        self.assertEqual(times, [0, 2.5])
        callDuration = 0
        clock.advance(0.5)
        self.assertEqual(times, [0, 2.5, 4])
        self.assertEqual(clock.seconds(), 4)

    def test_reactorTimeSkips(self):
        """
        When more time than the defined interval passes between when
        L{LoopingCall} schedules itself to run again and when it actually
        runs again, it should schedule the next call for the next interval
        which is still in the future.
        """
        times = []
        clock = task.Clock()

        def aCallback():
            times.append(clock.seconds())
        call = task.LoopingCall(aCallback)
        call.clock = clock
        call.start(0.5)
        self.assertEqual(times, [0])
        clock.advance(2)
        self.assertEqual(times, [0, 2])
        clock.advance(1)
        self.assertEqual(times, [0, 2, 3])
        clock.advance(0)
        self.assertEqual(times, [0, 2, 3])

    def test_reactorTimeCountSkips(self):
        """
        When L{LoopingCall} schedules itself to run again, if more than the
        specified interval has passed, it should schedule the next call for the
        next interval which is still in the future. If it was created
        using L{LoopingCall.withCount}, a positional argument will be
        inserted at the beginning of the argument list, indicating the number
        of calls that should have been made.
        """
        times = []
        clock = task.Clock()

        def aCallback(numCalls):
            times.append((clock.seconds(), numCalls))
        call = task.LoopingCall.withCount(aCallback)
        call.clock = clock
        INTERVAL = 0.5
        REALISTIC_DELAY = 0.01
        call.start(INTERVAL)
        self.assertEqual(times, [(0, 1)])
        clock.advance(INTERVAL + REALISTIC_DELAY)
        self.assertEqual(times, [(0, 1), (INTERVAL + REALISTIC_DELAY, 1)])
        clock.advance(3 * INTERVAL + REALISTIC_DELAY)
        self.assertEqual(times, [(0, 1), (INTERVAL + REALISTIC_DELAY, 1), (4 * INTERVAL + 2 * REALISTIC_DELAY, 3)])
        clock.advance(0)
        self.assertEqual(times, [(0, 1), (INTERVAL + REALISTIC_DELAY, 1), (4 * INTERVAL + 2 * REALISTIC_DELAY, 3)])

    def test_countLengthyIntervalCounts(self):
        """
        L{LoopingCall.withCount} counts only calls that were expected to be
        made.  So, if more than one, but less than two intervals pass between
        invocations, it won't increase the count above 1.  For example, a
        L{LoopingCall} with interval T expects to be invoked at T, 2T, 3T, etc.
        However, the reactor takes some time to get around to calling it, so in
        practice it will be called at T+something, 2T+something, 3T+something;
        and due to other things going on in the reactor, "something" is
        variable.  It won't increase the count unless "something" is greater
        than T.  So if the L{LoopingCall} is invoked at T, 2.75T, and 3T,
        the count has not increased, even though the distance between
        invocation 1 and invocation 2 is 1.75T.
        """
        times = []
        clock = task.Clock()

        def aCallback(count):
            times.append((clock.seconds(), count))
        call = task.LoopingCall.withCount(aCallback)
        call.clock = clock
        INTERVAL = 0.5
        REALISTIC_DELAY = 0.01
        call.start(INTERVAL)
        self.assertEqual(times.pop(), (0, 1))
        clock.advance(INTERVAL + REALISTIC_DELAY)
        self.assertEqual(times.pop(), (INTERVAL + REALISTIC_DELAY, 1))
        clock.advance(INTERVAL * 1.75)
        self.assertEqual(times.pop(), (2.75 * INTERVAL + REALISTIC_DELAY, 1))
        clock.advance(INTERVAL * 0.25)
        self.assertEqual(times.pop(), (3.0 * INTERVAL + REALISTIC_DELAY, 1))

    def test_withCountFloatingPointBoundary(self):
        """
        L{task.LoopingCall.withCount} should never invoke its callable with a
        zero.  Specifically, if a L{task.LoopingCall} created with C{withCount}
        has its L{start <task.LoopingCall.start>} method invoked with a
        floating-point number which introduces decimal inaccuracy when
        multiplied or divided, such as "0.1", L{task.LoopingCall} will never
        invoke its callable with 0.  Also, the sum of all the values passed to
        its callable as the "count" will be an integer, the number of intervals
        that have elapsed.

        This is a regression test for a particularly tricky case to implement.
        """
        clock = task.Clock()
        accumulator = []
        call = task.LoopingCall.withCount(accumulator.append)
        call.clock = clock
        count = 10
        timespan = 1.0
        interval = timespan / count
        call.start(interval, now=False)
        for x in range(count):
            clock.advance(interval)

        def sum_compat(items):
            """
            Make sure the result is more precise.
            On Python 3.11 or older this can be a float with ~ 0.00001
            in precision difference.
            See: https://github.com/python/cpython/issues/100425
            """
            total = 0.0
            for item in items:
                total += item
            return total
        epsilon = timespan - sum_compat([interval] * count)
        clock.advance(epsilon)
        secondsValue = clock.seconds()
        self.assertTrue(abs(epsilon) > 0.0, f'{epsilon} should be greater than zero')
        self.assertTrue(secondsValue >= timespan, f'{secondsValue} should be greater than or equal to {timespan}')
        self.assertEqual(sum_compat(accumulator), count)
        self.assertNotIn(0, accumulator)

    def test_withCountIntervalZero(self):
        """
        L{task.LoopingCall.withCount} with interval set to 0 calls the
        countCallable with a count of 1.
        """
        clock = task.Clock()
        accumulator = []

        def foo(cnt):
            accumulator.append(cnt)
            if len(accumulator) > 4:
                loop.stop()
        loop = task.LoopingCall.withCount(foo)
        loop.clock = clock
        deferred = loop.start(0, now=False)
        clock.pump([0] * 5)
        self.successResultOf(deferred)
        self.assertEqual([1] * 5, accumulator)

    def test_withCountIntervalZeroDelay(self):
        """
        L{task.LoopingCall.withCount} with interval set to 0 and a delayed
        call during the loop run will still call the countCallable 1 as if
        no delay occurred.
        """
        clock = task.Clock()
        deferred = defer.Deferred()
        accumulator = []

        def foo(cnt):
            accumulator.append(cnt)
            if len(accumulator) == 2:
                return deferred
            if len(accumulator) > 4:
                loop.stop()
        loop = task.LoopingCall.withCount(foo)
        loop.clock = clock
        loop.start(0, now=False)
        clock.pump([0] * 2)
        self.assertEqual([1] * 2, accumulator)
        clock.pump([1] * 2)
        self.assertEqual([1] * 2, accumulator)
        deferred.callback(None)
        clock.pump([0] * 4)
        self.assertEqual([1] * 5, accumulator)

    def test_withCountIntervalZeroDelayThenNonZeroInterval(self):
        """
        L{task.LoopingCall.withCount} with interval set to 0 will still keep
        the time when last called so when the interval is reset.
        """
        clock = task.Clock()
        deferred = defer.Deferred()
        accumulator = []
        stepsBeforeDelay = 2
        extraTimeAfterDelay = 5
        mutatedLoopInterval = 2
        durationOfDelay = 9
        skippedTime = extraTimeAfterDelay + durationOfDelay
        expectedSkipCount = skippedTime // mutatedLoopInterval
        expectedSkipCount += 1

        def foo(cnt):
            accumulator.append(cnt)
            if len(accumulator) == stepsBeforeDelay:
                return deferred
        loop = task.LoopingCall.withCount(foo)
        loop.clock = clock
        loop.start(0, now=False)
        clock.pump([1] * (stepsBeforeDelay + extraTimeAfterDelay))
        self.assertEqual([1] * stepsBeforeDelay, accumulator)
        loop.interval = mutatedLoopInterval
        deferred.callback(None)
        clock.advance(durationOfDelay)
        self.assertEqual([1, 1, expectedSkipCount], accumulator)
        clock.advance(1 * mutatedLoopInterval)
        self.assertEqual([1, 1, expectedSkipCount, 1], accumulator)
        clock.advance(2 * mutatedLoopInterval)
        self.assertEqual([1, 1, expectedSkipCount, 1, 2], accumulator)

    def testBasicFunction(self):
        timings = [0.05, 0.1, 0.1]
        clock = task.Clock()
        L = []

        def foo(a, b, c=None, d=None):
            L.append((a, b, c, d))
        lc = TestableLoopingCall(clock, foo, 'a', 'b', d='d')
        D = lc.start(0.1)
        theResult = []

        def saveResult(result):
            theResult.append(result)
        D.addCallback(saveResult)
        clock.pump(timings)
        self.assertEqual(len(L), 3, 'got %d iterations, not 3' % (len(L),))
        for a, b, c, d in L:
            self.assertEqual(a, 'a')
            self.assertEqual(b, 'b')
            self.assertIsNone(c)
            self.assertEqual(d, 'd')
        lc.stop()
        self.assertIs(theResult[0], lc)
        self.assertFalse(clock.calls)

    def testDelayedStart(self):
        timings = [0.05, 0.1, 0.1]
        clock = task.Clock()
        L = []
        lc = TestableLoopingCall(clock, L.append, None)
        d = lc.start(0.1, now=False)
        theResult = []

        def saveResult(result):
            theResult.append(result)
        d.addCallback(saveResult)
        clock.pump(timings)
        self.assertEqual(len(L), 2, 'got %d iterations, not 2' % (len(L),))
        lc.stop()
        self.assertIs(theResult[0], lc)
        self.assertFalse(clock.calls)

    def testBadDelay(self):
        lc = task.LoopingCall(lambda: None)
        self.assertRaises(ValueError, lc.start, -1)

    def _stoppingTest(self, delay):
        ran = []

        def foo():
            ran.append(None)
        clock = task.Clock()
        lc = TestableLoopingCall(clock, foo)
        lc.start(delay, now=False)
        lc.stop()
        self.assertFalse(ran)
        self.assertFalse(clock.calls)

    def testStopAtOnce(self):
        return self._stoppingTest(0)

    def testStoppingBeforeDelayedStart(self):
        return self._stoppingTest(10)

    def test_reset(self):
        """
        Test that L{LoopingCall} can be reset.
        """
        ran = []

        def foo():
            ran.append(None)
        c = task.Clock()
        lc = TestableLoopingCall(c, foo)
        lc.start(2, now=False)
        c.advance(1)
        lc.reset()
        c.advance(1)
        self.assertEqual(ran, [])
        c.advance(1)
        self.assertEqual(ran, [None])

    def test_reprFunction(self):
        """
        L{LoopingCall.__repr__} includes the wrapped function's name.
        """
        self.assertEqual(repr(task.LoopingCall(installReactor, 1, key=2)), "LoopingCall<None>(installReactor, *(1,), **{'key': 2})")

    def test_reprMethod(self):
        """
        L{LoopingCall.__repr__} includes the wrapped method's full name.
        """
        self.assertEqual(repr(task.LoopingCall(TestableLoopingCall.__init__)), 'LoopingCall<None>(TestableLoopingCall.__init__, *(), **{})')

    def test_deferredDeprecation(self):
        """
        L{LoopingCall.deferred} is deprecated.
        """
        loop = task.LoopingCall(lambda: None)
        loop.deferred
        message = 'twisted.internet.task.LoopingCall.deferred was deprecated in Twisted 16.0.0; please use the deferred returned by start() instead'
        warnings = self.flushWarnings([self.test_deferredDeprecation])
        self.assertEqual(1, len(warnings))
        self.assertEqual(DeprecationWarning, warnings[0]['category'])
        self.assertEqual(message, warnings[0]['message'])