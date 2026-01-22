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
class DeferredFilesystemLockTests(unittest.TestCase):
    """
    Test the behavior of L{DeferredFilesystemLock}
    """

    def setUp(self) -> None:
        self.clock = Clock()
        self.lock = DeferredFilesystemLock(self.mktemp(), scheduler=self.clock)

    def test_waitUntilLockedWithNoLock(self) -> Deferred[None]:
        """
        Test that the lock can be acquired when no lock is held
        """
        return self.lock.deferUntilLocked(timeout=1)

    def test_waitUntilLockedWithTimeoutLocked(self) -> Deferred[None]:
        """
        Test that the lock can not be acquired when the lock is held
        for longer than the timeout.
        """
        self.assertTrue(self.lock.lock())
        d = self.lock.deferUntilLocked(timeout=5.5)
        self.assertFailure(d, defer.TimeoutError)
        self.clock.pump([1] * 10)
        return d

    def test_waitUntilLockedWithTimeoutUnlocked(self) -> Deferred[None]:
        """
        Test that a lock can be acquired while a lock is held
        but the lock is unlocked before our timeout.
        """

        def onTimeout(f: Failure) -> None:
            f.trap(defer.TimeoutError)
            self.fail('Should not have timed out')
        self.assertTrue(self.lock.lock())
        self.clock.callLater(1, self.lock.unlock)
        d = self.lock.deferUntilLocked(timeout=10)
        d.addErrback(onTimeout)
        self.clock.pump([1] * 10)
        return d

    def test_defaultScheduler(self) -> None:
        """
        Test that the default scheduler is set up properly.
        """
        lock = DeferredFilesystemLock(self.mktemp())
        self.assertEqual(lock._scheduler, reactor)

    def test_concurrentUsage(self) -> Deferred[None]:
        """
        Test that an appropriate exception is raised when attempting
        to use deferUntilLocked concurrently.
        """
        self.lock.lock()
        self.clock.callLater(1, self.lock.unlock)
        d1 = self.lock.deferUntilLocked()
        d2 = self.lock.deferUntilLocked()
        self.assertFailure(d2, defer.AlreadyTryingToLockError)
        self.clock.advance(1)
        return d1

    def test_multipleUsages(self) -> Deferred[None]:
        """
        Test that a DeferredFilesystemLock can be used multiple times
        """

        def lockAquired(ign: object) -> Deferred[None]:
            self.lock.unlock()
            d = self.lock.deferUntilLocked()
            return d
        self.lock.lock()
        self.clock.callLater(1, self.lock.unlock)
        d = self.lock.deferUntilLocked()
        d.addCallback(lockAquired)
        self.clock.advance(1)
        return d

    def test_cancelDeferUntilLocked(self) -> None:
        """
        When cancelling a L{Deferred} returned by
        L{DeferredFilesystemLock.deferUntilLocked}, the
        L{DeferredFilesystemLock._tryLockCall} is cancelled.
        """
        self.lock.lock()
        deferred = self.lock.deferUntilLocked()
        tryLockCall = self.lock._tryLockCall
        assert tryLockCall is not None
        deferred.cancel()
        self.assertFalse(tryLockCall.active())
        self.assertIsNone(self.lock._tryLockCall)
        self.failureResultOf(deferred, defer.CancelledError)

    def test_cancelDeferUntilLockedWithTimeout(self) -> None:
        """
        When cancel a L{Deferred} returned by
        L{DeferredFilesystemLock.deferUntilLocked}, if the timeout is
        set, the timeout call will be cancelled.
        """
        self.lock.lock()
        deferred = self.lock.deferUntilLocked(timeout=1)
        timeoutCall = self.lock._timeoutCall
        assert timeoutCall is not None
        deferred.cancel()
        self.assertFalse(timeoutCall.active())
        self.assertIsNone(self.lock._timeoutCall)
        self.failureResultOf(deferred, defer.CancelledError)