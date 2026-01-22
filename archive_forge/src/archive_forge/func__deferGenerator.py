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
def _deferGenerator(g: _DeferableGenerator, deferred: Deferred[object]) -> Deferred[Any]:
    """
    See L{deferredGenerator}.
    """
    result = None
    waiting: List[Any] = [True, None]
    while 1:
        try:
            result = next(g)
        except StopIteration:
            deferred.callback(result)
            return deferred
        except BaseException:
            deferred.errback()
            return deferred
        if isinstance(result, Deferred):
            return fail(TypeError('Yield waitForDeferred(d), not d!'))
        if isinstance(result, waitForDeferred):

            def gotResult(r: object, result: waitForDeferred=cast(waitForDeferred, result)) -> None:
                result.result = r
                if waiting[0]:
                    waiting[0] = False
                    waiting[1] = r
                else:
                    _deferGenerator(g, deferred)
            result.d.addBoth(gotResult)
            if waiting[0]:
                waiting[0] = False
                return deferred
            waiting[0] = True
            waiting[1] = None
            result = None