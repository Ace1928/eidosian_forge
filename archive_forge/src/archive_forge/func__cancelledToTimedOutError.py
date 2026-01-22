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
def _cancelledToTimedOutError(value: _T, timeout: float) -> _T:
    """
    A default translation function that translates L{Failure}s that are
    L{CancelledError}s to L{TimeoutError}s.

    @param value: Anything
    @param timeout: The timeout

    @raise TimeoutError: If C{value} is a L{Failure} that is a L{CancelledError}.
    @raise Exception: If C{value} is a L{Failure} that is not a L{CancelledError},
        it is re-raised.

    @since: 16.5
    """
    if isinstance(value, Failure):
        value.trap(CancelledError)
        raise TimeoutError(timeout, 'Deferred')
    return value