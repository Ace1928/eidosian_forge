from __future__ import annotations
import inspect
import traceback
import warnings
from abc import ABC, abstractmethod
from asyncio import AbstractEventLoop, Future, iscoroutine
from contextvars import Context as _Context, copy_context as _copy_context
from enum import Enum
from functools import wraps
from sys import exc_info, implementation
from types import CoroutineType, GeneratorType, MappingProxyType, TracebackType
from typing import (
import attr
from incremental import Version
from typing_extensions import Concatenate, Literal, ParamSpec, Self
from twisted.internet.interfaces import IDelayedCall, IReactorTime
from twisted.logger import Logger
from twisted.python import lockfile
from twisted.python.compat import _PYPY, cmp, comparable
from twisted.python.deprecate import deprecated, warnAboutFunction
from twisted.python.failure import Failure, _extraneous
class DeferredLock(_ConcurrencyPrimitive):
    """
    A lock for event driven systems.

    @ivar locked: C{True} when this Lock has been acquired, false at all other
        times.  Do not change this value, but it is useful to examine for the
        equivalent of a "non-blocking" acquisition.
    """
    locked = False

    def _cancelAcquire(self: Self, d: Deferred[Self]) -> None:
        """
        Remove a deferred d from our waiting list, as the deferred has been
        canceled.

        Note: We do not need to wrap this in a try/except to catch d not
        being in self.waiting because this canceller will not be called if
        d has fired. release() pops a deferred out of self.waiting and
        calls it, so the canceller will no longer be called.

        @param d: The deferred that has been canceled.
        """
        self.waiting.remove(d)

    def acquire(self: Self) -> Deferred[Self]:
        """
        Attempt to acquire the lock.  Returns a L{Deferred} that fires on
        lock acquisition with the L{DeferredLock} as the value.  If the lock
        is locked, then the Deferred is placed at the end of a waiting list.

        @return: a L{Deferred} which fires on lock acquisition.
        @rtype: a L{Deferred}
        """
        d: Deferred[Self] = Deferred(canceller=self._cancelAcquire)
        if self.locked:
            self.waiting.append(d)
        else:
            self.locked = True
            d.callback(self)
        return d

    def release(self: Self) -> None:
        """
        Release the lock.  If there is a waiting list, then the first
        L{Deferred} in that waiting list will be called back.

        Should be called by whomever did the L{acquire}() when the shared
        resource is free.
        """
        assert self.locked, 'Tried to release an unlocked lock'
        self.locked = False
        if self.waiting:
            self.locked = True
            d = self.waiting.pop(0)
            d.callback(self)