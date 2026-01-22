from __future__ import annotations
import contextvars
import functools
import gc
import re
import traceback
import types
import unittest as pyunit
import warnings
import weakref
from asyncio import (
from typing import (
from hamcrest import assert_that, empty, equal_to
from hypothesis import given
from hypothesis.strategies import integers
from typing_extensions import assert_type
from twisted.internet import defer, reactor
from twisted.internet.defer import (
from twisted.internet.task import Clock
from twisted.python import log
from twisted.python.compat import _PYPY
from twisted.python.failure import Failure
from twisted.trial import unittest
class OtherPrimitivesTests(unittest.SynchronousTestCase, ImmediateFailureMixin):

    def _incr(self, result: object) -> None:
        self.counter += 1

    def setUp(self) -> None:
        self.counter = 0

    def testLock(self) -> None:
        lock = DeferredLock()
        lock.acquire().addCallback(self._incr)
        self.assertTrue(lock.locked)
        self.assertEqual(self.counter, 1)
        lock.acquire().addCallback(self._incr)
        self.assertTrue(lock.locked)
        self.assertEqual(self.counter, 1)
        lock.release()
        self.assertTrue(lock.locked)
        self.assertEqual(self.counter, 2)
        lock.release()
        self.assertFalse(lock.locked)
        self.assertEqual(self.counter, 2)
        self.assertRaises(TypeError, lock.run)
        firstUnique = object()
        secondUnique = object()
        controlDeferred: Deferred[object] = Deferred()
        result: Optional[object] = None

        def helper(resultValue: object, returnValue: object=None) -> object:
            nonlocal result
            result = resultValue
            return returnValue
        resultDeferred = lock.run(helper, resultValue=firstUnique, returnValue=controlDeferred)
        self.assertTrue(lock.locked)
        self.assertEqual(result, firstUnique)
        resultDeferred.addCallback(helper)
        lock.acquire().addCallback(self._incr)
        self.assertTrue(lock.locked)
        self.assertEqual(self.counter, 2)
        controlDeferred.callback(secondUnique)
        self.assertEqual(result, secondUnique)
        self.assertTrue(lock.locked)
        self.assertEqual(self.counter, 3)
        d = lock.acquire().addBoth(helper)
        d.cancel()
        self.assertIsInstance(result, Failure)
        self.assertEqual(cast(Failure, result).type, defer.CancelledError)
        lock.release()
        self.assertFalse(lock.locked)

        def returnsInt() -> Deferred[int]:
            return defer.succeed(2)

        async def returnsCoroInt() -> int:
            return 1
        assert_type(lock.run(returnsInt), Deferred[int])
        assert_type(lock.run(returnsCoroInt), Deferred[int])

    def test_cancelLockAfterAcquired(self) -> None:
        """
        When canceling a L{Deferred} from a L{DeferredLock} that already
        has the lock, the cancel should have no effect.
        """

        def failOnErrback(f: Failure) -> None:
            self.fail('Unexpected errback call!')
        lock = DeferredLock()
        d = lock.acquire()
        d.addErrback(failOnErrback)
        d.cancel()

    def test_cancelLockBeforeAcquired(self) -> None:
        """
        When canceling a L{Deferred} from a L{DeferredLock} that does not
        yet have the lock (i.e., the L{Deferred} has not fired), the cancel
        should cause a L{defer.CancelledError} failure.
        """
        lock = DeferredLock()
        lock.acquire()
        d = lock.acquire()
        d.cancel()
        self.assertImmediateFailure(d, defer.CancelledError)

    def testSemaphore(self) -> None:
        N = 13
        sem = DeferredSemaphore(N)
        controlDeferred: Deferred[None] = Deferred()
        helperArg: object = None

        def helper(arg: object) -> Deferred[None]:
            nonlocal helperArg
            helperArg = arg
            return controlDeferred
        results: List[object] = []
        uniqueObject = object()
        resultDeferred = sem.run(helper, arg=uniqueObject)
        resultDeferred.addCallback(results.append)
        resultDeferred.addCallback(self._incr)
        self.assertEqual(results, [])
        self.assertEqual(helperArg, uniqueObject)
        controlDeferred.callback(None)
        self.assertIsNone(results.pop())
        self.assertEqual(self.counter, 1)
        self.counter = 0
        for i in range(1, 1 + N):
            sem.acquire().addCallback(self._incr)
            self.assertEqual(self.counter, i)
        success = []

        def fail(r: object) -> None:
            success.append(False)

        def succeed(r: object) -> None:
            success.append(True)
        d = sem.acquire().addCallbacks(fail, succeed)
        d.cancel()
        self.assertEqual(success, [True])
        sem.acquire().addCallback(self._incr)
        self.assertEqual(self.counter, N)
        sem.release()
        self.assertEqual(self.counter, N + 1)
        for i in range(1, 1 + N):
            sem.release()
            self.assertEqual(self.counter, N + 1)

    def test_semaphoreInvalidTokens(self) -> None:
        """
        If the token count passed to L{DeferredSemaphore} is less than one
        then L{ValueError} is raised.
        """
        self.assertRaises(ValueError, DeferredSemaphore, 0)
        self.assertRaises(ValueError, DeferredSemaphore, -1)

    def test_cancelSemaphoreAfterAcquired(self) -> None:
        """
        When canceling a L{Deferred} from a L{DeferredSemaphore} that
        already has the semaphore, the cancel should have no effect.
        """

        def failOnErrback(f: Failure) -> None:
            self.fail('Unexpected errback call!')
        sem = DeferredSemaphore(1)
        d = sem.acquire()
        d.addErrback(failOnErrback)
        d.cancel()

    def test_cancelSemaphoreBeforeAcquired(self) -> None:
        """
        When canceling a L{Deferred} from a L{DeferredSemaphore} that does
        not yet have the semaphore (i.e., the L{Deferred} has not fired),
        the cancel should cause a L{defer.CancelledError} failure.
        """
        sem = DeferredSemaphore(1)
        sem.acquire()
        d = sem.acquire()
        d.cancel()
        self.assertImmediateFailure(d, defer.CancelledError)

    def testQueue(self) -> None:
        N, M = (2, 2)
        queue: DeferredQueue[int] = DeferredQueue(N, M)
        gotten: List[int] = []
        for i in range(M):
            queue.get().addCallback(gotten.append)
        self.assertRaises(defer.QueueUnderflow, queue.get)
        for i in range(M):
            queue.put(i)
            self.assertEqual(gotten, list(range(i + 1)))
        for i in range(N):
            queue.put(N + i)
            self.assertEqual(gotten, list(range(M)))
        self.assertRaises(defer.QueueOverflow, queue.put, None)
        gotten = []
        for i in range(N):
            queue.get().addCallback(gotten.append)
            self.assertEqual(gotten, list(range(N, N + i + 1)))
        queue = DeferredQueue()
        gotten = []
        for i in range(N):
            queue.get().addCallback(gotten.append)
        for i in range(N):
            queue.put(i)
        self.assertEqual(gotten, list(range(N)))
        queue = DeferredQueue(size=0)
        self.assertRaises(defer.QueueOverflow, queue.put, None)
        queue = DeferredQueue(backlog=0)
        self.assertRaises(defer.QueueUnderflow, queue.get)

    def test_cancelQueueAfterSynchronousGet(self) -> None:
        """
        When canceling a L{Deferred} from a L{DeferredQueue} that already has
        a result, the cancel should have no effect.
        """

        def failOnErrback(f: Failure) -> None:
            self.fail('Unexpected errback call!')
        queue: DeferredQueue[None] = DeferredQueue()
        d = queue.get()
        d.addErrback(failOnErrback)
        queue.put(None)
        d.cancel()

    def test_cancelQueueAfterGet(self) -> None:
        """
        When canceling a L{Deferred} from a L{DeferredQueue} that does not
        have a result (i.e., the L{Deferred} has not fired), the cancel
        causes a L{defer.CancelledError} failure. If the queue has a result
        later on, it doesn't try to fire the deferred.
        """
        queue: DeferredQueue[None] = DeferredQueue()
        d = queue.get()
        d.cancel()
        self.assertImmediateFailure(d, defer.CancelledError)

        def cb(ignore: object) -> Deferred[None]:
            queue.put(None)
            return queue.get().addCallback(self.assertIs, None)
        d.addCallback(cb)
        done: List[None] = []
        d.addCallback(done.append)
        self.assertEqual(len(done), 1)