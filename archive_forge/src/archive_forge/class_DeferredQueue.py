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
class DeferredQueue(Generic[_T]):
    """
    An event driven queue.

    Objects may be added as usual to this queue.  When an attempt is
    made to retrieve an object when the queue is empty, a L{Deferred} is
    returned which will fire when an object becomes available.

    @ivar size: The maximum number of objects to allow into the queue
        at a time.  When an attempt to add a new object would exceed this
        limit, L{QueueOverflow} is raised synchronously.  L{None} for no limit.
    @ivar backlog: The maximum number of L{Deferred} gets to allow at
        one time.  When an attempt is made to get an object which would
        exceed this limit, L{QueueUnderflow} is raised synchronously.  L{None}
        for no limit.
    """

    def __init__(self, size: Optional[int]=None, backlog: Optional[int]=None) -> None:
        self.waiting: List[Deferred[_T]] = []
        self.pending: List[_T] = []
        self.size = size
        self.backlog = backlog

    def _cancelGet(self, d: Deferred[_T]) -> None:
        """
        Remove a deferred d from our waiting list, as the deferred has been
        canceled.

        Note: We do not need to wrap this in a try/except to catch d not
        being in self.waiting because this canceller will not be called if
        d has fired. put() pops a deferred out of self.waiting and calls
        it, so the canceller will no longer be called.

        @param d: The deferred that has been canceled.
        """
        self.waiting.remove(d)

    def put(self, obj: _T) -> None:
        """
        Add an object to this queue.

        @raise QueueOverflow: Too many objects are in this queue.
        """
        if self.waiting:
            self.waiting.pop(0).callback(obj)
        elif self.size is None or len(self.pending) < self.size:
            self.pending.append(obj)
        else:
            raise QueueOverflow()

    def get(self) -> Deferred[_T]:
        """
        Attempt to retrieve and remove an object from the queue.

        @return: a L{Deferred} which fires with the next object available in
        the queue.

        @raise QueueUnderflow: Too many (more than C{backlog})
        L{Deferred}s are already waiting for an object from this queue.
        """
        if self.pending:
            return succeed(self.pending.pop(0))
        elif self.backlog is None or len(self.waiting) < self.backlog:
            d: Deferred[_T] = Deferred(canceller=self._cancelGet)
            self.waiting.append(d)
            return d
        else:
            raise QueueUnderflow()