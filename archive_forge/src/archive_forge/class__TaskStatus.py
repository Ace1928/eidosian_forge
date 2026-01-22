from __future__ import annotations
import enum
import functools
import gc
import itertools
import random
import select
import sys
import threading
import warnings
from collections import deque
from contextlib import AbstractAsyncContextManager, contextmanager, suppress
from contextvars import copy_context
from heapq import heapify, heappop, heappush
from math import inf
from time import perf_counter
from typing import (
import attrs
from outcome import Error, Outcome, Value, capture
from sniffio import thread_local as sniffio_library
from sortedcontainers import SortedDict
from .. import _core
from .._abc import Clock, Instrument
from .._deprecate import warn_deprecated
from .._util import NoPublicConstructor, coroutine_or_error, final
from ._asyncgens import AsyncGenerators
from ._concat_tb import concat_tb
from ._entry_queue import EntryQueue, TrioToken
from ._exceptions import Cancelled, RunFinishedError, TrioInternalError
from ._instrumentation import Instruments
from ._ki import LOCALS_KEY_KI_PROTECTION_ENABLED, KIManager, enable_ki_protection
from ._thread_cache import start_thread_soon
from ._traps import (
from ._generated_instrumentation import *
from ._generated_run import *
@attrs.define(eq=False, hash=False, repr=False, slots=False)
class _TaskStatus(TaskStatus[StatusT]):
    _old_nursery: Nursery
    _new_nursery: Nursery
    _value: StatusT | type[_NoStatus] = _NoStatus

    def __repr__(self) -> str:
        return f'<Task status object at {id(self):#x}>'

    @overload
    def started(self: _TaskStatus[None]) -> None:
        ...

    @overload
    def started(self: _TaskStatus[StatusT], value: StatusT) -> None:
        ...

    def started(self, value: StatusT | None=None) -> None:
        if self._value is not _NoStatus:
            raise RuntimeError("called 'started' twice on the same task status")
        self._value = cast(StatusT, value)
        assert self._old_nursery._cancel_status is not None
        if self._old_nursery._cancel_status.effectively_cancelled:
            return
        assert not self._new_nursery._closed
        tasks = self._old_nursery._children
        self._old_nursery._children = set()
        for task in tasks:
            task._parent_nursery = self._new_nursery
            task._eventual_parent_nursery = None
            self._new_nursery._children.add(task)
        cancel_status_children = self._old_nursery._cancel_status.children
        cancel_status_tasks = set(self._old_nursery._cancel_status.tasks)
        cancel_status_tasks.discard(self._old_nursery._parent_task)
        for cancel_status in cancel_status_children:
            cancel_status.parent = None
        for task in cancel_status_tasks:
            task._activate_cancel_status(None)
        for cancel_status in cancel_status_children:
            cancel_status.parent = self._new_nursery._cancel_status
        for task in cancel_status_tasks:
            task._activate_cancel_status(self._new_nursery._cancel_status)
        assert not self._old_nursery._children
        self._old_nursery._check_nursery_closed()