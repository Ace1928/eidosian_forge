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
class DebugInfo:
    """
    Deferred debug helper.
    """
    failResult: Optional[Failure] = None
    creator: Optional[List[str]] = None
    invoker: Optional[List[str]] = None

    def _getDebugTracebacks(self) -> str:
        info = ''
        if self.creator is not None:
            info += ' C: Deferred was created:\n C:'
            info += ''.join(self.creator).rstrip().replace('\n', '\n C:')
            info += '\n'
        if self.invoker is not None:
            info += ' I: First Invoker was:\n I:'
            info += ''.join(self.invoker).rstrip().replace('\n', '\n I:')
            info += '\n'
        return info

    def __del__(self) -> None:
        """
        Print tracebacks and die.

        If the *last* (and I do mean *last*) callback leaves me in an error
        state, print a traceback (if said errback is a L{Failure}).
        """
        if self.failResult is not None:
            log.critical('Unhandled error in Deferred:', isError=True)
            debugInfo = self._getDebugTracebacks()
            if debugInfo:
                format = '(debug: {debugInfo})'
            else:
                format = ''
            log.failure(format, self.failResult, debugInfo=debugInfo)