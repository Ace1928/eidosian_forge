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
@final
class Nursery(metaclass=NoPublicConstructor):
    """A context which may be used to spawn (or cancel) child tasks.

    Not constructed directly, use `open_nursery` instead.

    The nursery will remain open until all child tasks have completed,
    or until it is cancelled, at which point it will cancel all its
    remaining child tasks and close.

    Nurseries ensure the absence of orphaned Tasks, since all running
    tasks will belong to an open Nursery.

    Attributes:
        cancel_scope:
            Creating a nursery also implicitly creates a cancellation scope,
            which is exposed as the :attr:`cancel_scope` attribute. This is
            used internally to implement the logic where if an error occurs
            then ``__aexit__`` cancels all children, but you can use it for
            other things, e.g. if you want to explicitly cancel all children
            in response to some external event.
    """

    def __init__(self, parent_task: Task, cancel_scope: CancelScope, strict_exception_groups: bool):
        self._parent_task = parent_task
        self._strict_exception_groups = strict_exception_groups
        parent_task._child_nurseries.append(self)
        self._cancel_status = parent_task._cancel_status
        self.cancel_scope = cancel_scope
        assert self.cancel_scope._cancel_status is self._cancel_status
        self._children: set[Task] = set()
        self._pending_excs: list[BaseException] = []
        self._nested_child_running = True
        self._parent_waiting_in_aexit = False
        self._pending_starts = 0
        self._closed = False

    @property
    def child_tasks(self) -> frozenset[Task]:
        """(`frozenset`): Contains all the child :class:`~trio.lowlevel.Task`
        objects which are still running."""
        return frozenset(self._children)

    @property
    def parent_task(self) -> Task:
        """(`~trio.lowlevel.Task`):  The Task that opened this nursery."""
        return self._parent_task

    def _add_exc(self, exc: BaseException) -> None:
        self._pending_excs.append(exc)
        self.cancel_scope.cancel()

    def _check_nursery_closed(self) -> None:
        if not any([self._nested_child_running, self._children, self._pending_starts]):
            self._closed = True
            if self._parent_waiting_in_aexit:
                self._parent_waiting_in_aexit = False
                GLOBAL_RUN_CONTEXT.runner.reschedule(self._parent_task)

    def _child_finished(self, task: Task, outcome: Outcome[Any]) -> None:
        self._children.remove(task)
        if isinstance(outcome, Error):
            self._add_exc(outcome.error)
        self._check_nursery_closed()

    async def _nested_child_finished(self, nested_child_exc: BaseException | None) -> BaseException | None:
        if nested_child_exc is not None:
            self._add_exc(nested_child_exc)
        self._nested_child_running = False
        self._check_nursery_closed()
        if not self._closed:

            def aborted(raise_cancel: _core.RaiseCancelT) -> Abort:
                exn = capture(raise_cancel).error
                if not isinstance(exn, Cancelled):
                    self._add_exc(exn)
                del exn
                return Abort.FAILED
            self._parent_waiting_in_aexit = True
            await wait_task_rescheduled(aborted)
        else:
            try:
                await cancel_shielded_checkpoint()
            except BaseException as exc:
                self._add_exc(exc)
        popped = self._parent_task._child_nurseries.pop()
        assert popped is self
        if self._pending_excs:
            try:
                if not self._strict_exception_groups and len(self._pending_excs) == 1:
                    return self._pending_excs[0]
                exception = BaseExceptionGroup('Exceptions from Trio nursery', self._pending_excs)
                if not self._strict_exception_groups:
                    exception.add_note(NONSTRICT_EXCEPTIONGROUP_NOTE)
                return exception
            finally:
                del self._pending_excs
        return None

    def start_soon(self, async_fn: Callable[[Unpack[PosArgT]], Awaitable[object]], *args: Unpack[PosArgT], name: object=None) -> None:
        """Creates a child task, scheduling ``await async_fn(*args)``.

        If you want to run a function and immediately wait for its result,
        then you don't need a nursery; just use ``await async_fn(*args)``.
        If you want to wait for the task to initialize itself before
        continuing, see :meth:`start`, the other fundamental method for
        creating concurrent tasks in Trio.

        Note that this is *not* an async function and you don't use await
        when calling it. It sets up the new task, but then returns
        immediately, *before* the new task has a chance to do anything.
        New tasks may start running in any order, and at any checkpoint the
        scheduler chooses - at latest when the nursery is waiting to exit.

        It's possible to pass a nursery object into another task, which
        allows that task to start new child tasks in the first task's
        nursery.

        The child task inherits its parent nursery's cancel scopes.

        Args:
            async_fn: An async callable.
            args: Positional arguments for ``async_fn``. If you want
                  to pass keyword arguments, use
                  :func:`functools.partial`.
            name: The name for this task. Only used for
                  debugging/introspection
                  (e.g. ``repr(task_obj)``). If this isn't a string,
                  :meth:`start_soon` will try to make it one. A
                  common use case is if you're wrapping a function
                  before spawning a new task, you might pass the
                  original function as the ``name=`` to make
                  debugging easier.

        Raises:
            RuntimeError: If this nursery is no longer open
                          (i.e. its ``async with`` block has
                          exited).
        """
        GLOBAL_RUN_CONTEXT.runner.spawn_impl(async_fn, args, self, name)

    async def start(self, async_fn: Callable[..., Awaitable[object]], *args: object, name: object=None) -> Any:
        """Creates and initializes a child task.

        Like :meth:`start_soon`, but blocks until the new task has
        finished initializing itself, and optionally returns some
        information from it.

        The ``async_fn`` must accept a ``task_status`` keyword argument,
        and it must make sure that it (or someone) eventually calls
        :meth:`task_status.started() <TaskStatus.started>`.

        The conventional way to define ``async_fn`` is like::

            async def async_fn(arg1, arg2, *, task_status=trio.TASK_STATUS_IGNORED):
                ...  # Caller is blocked waiting for this code to run
                task_status.started()
                ...  # This async code can be interleaved with the caller

        :attr:`trio.TASK_STATUS_IGNORED` is a special global object with
        a do-nothing ``started`` method. This way your function supports
        being called either like ``await nursery.start(async_fn, arg1,
        arg2)`` or directly like ``await async_fn(arg1, arg2)``, and
        either way it can call :meth:`task_status.started() <TaskStatus.started>`
        without worrying about which mode it's in. Defining your function like
        this will make it obvious to readers that it supports being used
        in both modes.

        Before the child calls :meth:`task_status.started() <TaskStatus.started>`,
        it's effectively run underneath the call to :meth:`start`: if it
        raises an exception then that exception is reported by
        :meth:`start`, and does *not* propagate out of the nursery. If
        :meth:`start` is cancelled, then the child task is also
        cancelled.

        When the child calls :meth:`task_status.started() <TaskStatus.started>`,
        it's moved out from underneath :meth:`start` and into the given nursery.

        If the child task passes a value to :meth:`task_status.started(value) <TaskStatus.started>`,
        then :meth:`start` returns this value. Otherwise, it returns ``None``.
        """
        if self._closed:
            raise RuntimeError('Nursery is closed to new arrivals')
        try:
            self._pending_starts += 1
            try:
                async with open_nursery(strict_exception_groups=True) as old_nursery:
                    task_status: _TaskStatus[Any] = _TaskStatus(old_nursery, self)
                    thunk = functools.partial(async_fn, task_status=task_status)
                    task = GLOBAL_RUN_CONTEXT.runner.spawn_impl(thunk, args, old_nursery, name)
                    task._eventual_parent_nursery = self
            except BaseExceptionGroup as exc:
                if len(exc.exceptions) == 1:
                    raise exc.exceptions[0] from None
                raise TrioInternalError('Internal nursery should not have multiple tasks. This can be caused by the user managing to access the "old" nursery in `task_status` and spawning tasks in it.') from exc
            if task_status._value is _NoStatus:
                raise RuntimeError('child exited without calling task_status.started()')
            return task_status._value
        finally:
            self._pending_starts -= 1
            self._check_nursery_closed()

    def __del__(self) -> None:
        assert not self._children