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
def addTimeout(self, timeout: float, clock: IReactorTime, onTimeoutCancel: Optional[Callable[[Union[_SelfResultT, Failure], float], Union[_NextResultT, Failure]]]=None) -> 'Deferred[Union[_SelfResultT, _NextResultT]]':
    """
        Time out this L{Deferred} by scheduling it to be cancelled after
        C{timeout} seconds.

        The timeout encompasses all the callbacks and errbacks added to this
        L{defer.Deferred} before the call to L{addTimeout}, and none added
        after the call.

        If this L{Deferred} gets timed out, it errbacks with a L{TimeoutError},
        unless a cancelable function was passed to its initialization or unless
        a different C{onTimeoutCancel} callable is provided.

        @param timeout: number of seconds to wait before timing out this
            L{Deferred}
        @param clock: The object which will be used to schedule the timeout.
        @param onTimeoutCancel: A callable which is called immediately after
            this L{Deferred} times out, and not if this L{Deferred} is
            otherwise cancelled before the timeout. It takes an arbitrary
            value, which is the value of this L{Deferred} at that exact point
            in time (probably a L{CancelledError} L{Failure}), and the
            C{timeout}.  The default callable (if C{None} is provided) will
            translate a L{CancelledError} L{Failure} into a L{TimeoutError}.

        @return: C{self}.

        @since: 16.5
        """
    timedOut = [False]

    def timeItOut() -> None:
        timedOut[0] = True
        self.cancel()
    delayedCall = clock.callLater(timeout, timeItOut)

    def convertCancelled(result: Union[_SelfResultT, Failure]) -> Union[_SelfResultT, _NextResultT, Failure]:
        if timedOut[0]:
            toCall = onTimeoutCancel or _cancelledToTimedOutError
            return toCall(result, timeout)
        return result

    def cancelTimeout(result: _T) -> _T:
        if delayedCall.active():
            delayedCall.cancel()
        return result
    converted: Deferred[Union[_SelfResultT, _NextResultT]] = self.addBoth(convertCancelled)
    return converted.addBoth(cancelTimeout)