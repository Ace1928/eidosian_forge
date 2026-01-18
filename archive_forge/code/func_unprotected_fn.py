from __future__ import annotations
import contextlib
import contextvars
import inspect
import queue as stdlib_queue
import threading
from itertools import count
from typing import TYPE_CHECKING, Generic, TypeVar, overload
import attrs
import outcome
from attrs import define
from sniffio import current_async_library_cvar
import trio
from ._core import (
from ._deprecate import warn_deprecated
from ._sync import CapacityLimiter, Event
from ._util import coroutine_or_error
@disable_ki_protection
def unprotected_fn(self) -> RetT:
    ret = self.context.run(self.fn, *self.args)
    if inspect.iscoroutine(ret):
        ret.close()
        raise TypeError('Trio expected a synchronous function, but {!r} appears to be asynchronous'.format(getattr(self.fn, '__qualname__', self.fn)))
    return ret