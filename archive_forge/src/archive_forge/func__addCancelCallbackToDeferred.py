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
def _addCancelCallbackToDeferred(it: Deferred[_T], status: _CancellationStatus[_T]) -> None:
    """
    Helper for L{_cancellableInlineCallbacks} to add
    L{_handleCancelInlineCallbacks} as the first errback.

    @param it: The L{Deferred} to add the errback to.
    @param status: a L{_CancellationStatus} tracking the current status of C{gen}
    """
    it.callbacks, tmp = ([], it.callbacks)
    it = it.addErrback(_handleCancelInlineCallbacks, status)
    it.callbacks.extend(tmp)
    it.errback(_InternalInlineCallbacksCancelledError())