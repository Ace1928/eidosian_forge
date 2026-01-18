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
@wraps(f)
def unwindGenerator(*args: _P.args, **kwargs: _P.kwargs) -> Deferred[_T]:
    try:
        gen = f(*args, **kwargs)
    except _DefGen_Return:
        raise TypeError('inlineCallbacks requires %r to produce a generator; insteadcaught returnValue being used in a non-generator' % (f,))
    if not isinstance(gen, GeneratorType):
        raise TypeError('inlineCallbacks requires %r to produce a generator; instead got %r' % (f, gen))
    return _cancellableInlineCallbacks(gen)