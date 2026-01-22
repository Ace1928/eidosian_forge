import os
import sys
import time
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, threads
from twisted.python import failure, log, threadable, threadpool
from twisted.trial.unittest import TestCase
import time
import %(reactor)s
from twisted.internet import reactor
@skipIf(not interfaces.IReactorThreads(reactor, None), 'No thread support, nothing to test here.')
class ReactorThreadsTests(TestCase):
    """
    Tests for the reactor threading API.
    """

    def test_suggestThreadPoolSize(self):
        """
        Try to change maximum number of threads.
        """
        reactor.suggestThreadPoolSize(34)
        self.assertEqual(reactor.threadpool.max, 34)
        reactor.suggestThreadPoolSize(4)
        self.assertEqual(reactor.threadpool.max, 4)

    def _waitForThread(self):
        """
        The reactor's threadpool is only available when the reactor is running,
        so to have a sane behavior during the tests we make a dummy
        L{threads.deferToThread} call.
        """
        return threads.deferToThread(time.sleep, 0)

    def test_callInThread(self):
        """
        Test callInThread functionality: set a C{threading.Event}, and check
        that it's not in the main thread.
        """

        def cb(ign):
            waiter = threading.Event()
            result = []

            def threadedFunc():
                result.append(threadable.isInIOThread())
                waiter.set()
            reactor.callInThread(threadedFunc)
            waiter.wait(120)
            if not waiter.isSet():
                self.fail('Timed out waiting for event.')
            else:
                self.assertEqual(result, [False])
        return self._waitForThread().addCallback(cb)

    def test_callFromThread(self):
        """
        Test callFromThread functionality: from the main thread, and from
        another thread.
        """

        def cb(ign):
            firedByReactorThread = defer.Deferred()
            firedByOtherThread = defer.Deferred()

            def threadedFunc():
                reactor.callFromThread(firedByOtherThread.callback, None)
            reactor.callInThread(threadedFunc)
            reactor.callFromThread(firedByReactorThread.callback, None)
            return defer.DeferredList([firedByReactorThread, firedByOtherThread], fireOnOneErrback=True)
        return self._waitForThread().addCallback(cb)

    def test_wakerOverflow(self):
        """
        Try to make an overflow on the reactor waker using callFromThread.
        """

        def cb(ign):
            self.failure = None
            waiter = threading.Event()

            def threadedFunction():
                for i in range(100000):
                    try:
                        reactor.callFromThread(lambda: None)
                    except BaseException:
                        self.failure = failure.Failure()
                        break
                waiter.set()
            reactor.callInThread(threadedFunction)
            waiter.wait(120)
            if not waiter.isSet():
                self.fail('Timed out waiting for event')
            if self.failure is not None:
                return defer.fail(self.failure)
        return self._waitForThread().addCallback(cb)

    def _testBlockingCallFromThread(self, reactorFunc):
        """
        Utility method to test L{threads.blockingCallFromThread}.
        """
        waiter = threading.Event()
        results = []
        errors = []

        def cb1(ign):

            def threadedFunc():
                try:
                    r = threads.blockingCallFromThread(reactor, reactorFunc)
                except Exception as e:
                    errors.append(e)
                else:
                    results.append(r)
                waiter.set()
            reactor.callInThread(threadedFunc)
            return threads.deferToThread(waiter.wait, self.getTimeout())

        def cb2(ign):
            if not waiter.isSet():
                self.fail('Timed out waiting for event')
            return (results, errors)
        return self._waitForThread().addCallback(cb1).addBoth(cb2)

    def test_blockingCallFromThread(self):
        """
        Test blockingCallFromThread facility: create a thread, call a function
        in the reactor using L{threads.blockingCallFromThread}, and verify the
        result returned.
        """

        def reactorFunc():
            return defer.succeed('foo')

        def cb(res):
            self.assertEqual(res[0][0], 'foo')
        return self._testBlockingCallFromThread(reactorFunc).addCallback(cb)

    def test_asyncBlockingCallFromThread(self):
        """
        Test blockingCallFromThread as above, but be sure the resulting
        Deferred is not already fired.
        """

        def reactorFunc():
            d = defer.Deferred()
            reactor.callLater(0.1, d.callback, 'egg')
            return d

        def cb(res):
            self.assertEqual(res[0][0], 'egg')
        return self._testBlockingCallFromThread(reactorFunc).addCallback(cb)

    def test_errorBlockingCallFromThread(self):
        """
        Test error report for blockingCallFromThread.
        """

        def reactorFunc():
            return defer.fail(RuntimeError('bar'))

        def cb(res):
            self.assertIsInstance(res[1][0], RuntimeError)
            self.assertEqual(res[1][0].args[0], 'bar')
        return self._testBlockingCallFromThread(reactorFunc).addCallback(cb)

    def test_asyncErrorBlockingCallFromThread(self):
        """
        Test error report for blockingCallFromThread as above, but be sure the
        resulting Deferred is not already fired.
        """

        def reactorFunc():
            d = defer.Deferred()
            reactor.callLater(0.1, d.errback, RuntimeError('spam'))
            return d

        def cb(res):
            self.assertIsInstance(res[1][0], RuntimeError)
            self.assertEqual(res[1][0].args[0], 'spam')
        return self._testBlockingCallFromThread(reactorFunc).addCallback(cb)