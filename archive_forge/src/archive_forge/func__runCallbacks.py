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
def _runCallbacks(self) -> None:
    """
        Run the chain of callbacks once a result is available.

        This consists of a simple loop over all of the callbacks, calling each
        with the current result and making the current result equal to the
        return value (or raised exception) of that call.

        If L{_runningCallbacks} is true, this loop won't run at all, since
        it is already running above us on the call stack.  If C{self.paused} is
        true, the loop also won't run, because that's what it means to be
        paused.

        The loop will terminate before processing all of the callbacks if a
        L{Deferred} without a result is encountered.

        If a L{Deferred} I{with} a result is encountered, that result is taken
        and the loop proceeds.

        @note: The implementation is complicated slightly by the fact that
            chaining (associating two L{Deferred}s with each other such that one
            will wait for the result of the other, as happens when a Deferred is
            returned from a callback on another L{Deferred}) is supported
            iteratively rather than recursively, to avoid running out of stack
            frames when processing long chains.
        """
    if self._runningCallbacks:
        return
    chain: List[Deferred[Any]] = [self]
    while chain:
        current = chain[-1]
        if current.paused:
            return
        finished = True
        current._chainedTo = None
        while current.callbacks:
            item = current.callbacks.pop(0)
            if not isinstance(current.result, Failure):
                callback, args, kwargs = item[0]
            else:
                callback, args, kwargs = item[1]
            if callback is _CONTINUE:
                chainee = cast(Deferred[object], args[0])
                chainee.result = current.result
                current.result = None
                if current._debugInfo is not None:
                    current._debugInfo.failResult = None
                chainee.paused -= 1
                chain.append(chainee)
                finished = False
                break
            try:
                current._runningCallbacks = True
                try:
                    current.result = callback(current.result, *args, **kwargs)
                    if current.result is current:
                        warnAboutFunction(callback, 'Callback returned the Deferred it was attached to; this breaks the callback chain and will raise an exception in the future.')
                finally:
                    current._runningCallbacks = False
            except BaseException:
                current.result = Failure(captureVars=self.debug)
            else:
                if isinstance(current.result, Deferred):
                    resultResult = getattr(current.result, 'result', _NO_RESULT)
                    if resultResult is _NO_RESULT or isinstance(resultResult, Deferred) or current.result.paused:
                        current.pause()
                        current._chainedTo = current.result
                        current.result.callbacks.append(current._continuation())
                        break
                    else:
                        current.result.result = None
                        if current.result._debugInfo is not None:
                            current.result._debugInfo.failResult = None
                        current.result = resultResult
        if finished:
            if isinstance(current.result, Failure):
                current.result.cleanFailure()
                if current._debugInfo is None:
                    current._debugInfo = DebugInfo()
                current._debugInfo.failResult = current.result
            elif current._debugInfo is not None:
                current._debugInfo.failResult = None
            chain.pop()