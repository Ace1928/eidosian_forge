from __future__ import annotations
import contextvars
import functools
import gc
import sys
import threading
import time
import types
import weakref
from contextlib import ExitStack, contextmanager, suppress
from math import inf
from typing import TYPE_CHECKING, Any, NoReturn, TypeVar, cast
import outcome
import pytest
import sniffio
from ... import _core
from ..._threads import to_thread_run_sync
from ..._timeouts import fail_after, sleep
from ...testing import (
from .._run import DEADLINE_HEAP_MIN_PRUNE_THRESHOLD
from .tutil import (
@pytest.mark.filterwarnings('ignore:.*strict_exception_groups=False:trio.TrioDeprecationWarning')
@pytest.mark.parametrize('run_strict', [False, True])
@pytest.mark.parametrize('start_raiser_strict', [False, True, None])
@pytest.mark.parametrize('raise_after_started', [False, True])
@pytest.mark.parametrize('raise_custom_exc_grp', [False, True])
def test_trio_run_strict_before_started(run_strict: bool, start_raiser_strict: bool | None, raise_after_started: bool, raise_custom_exc_grp: bool) -> None:
    """
    Regression tests for #2611, where exceptions raised before
    `TaskStatus.started()` caused `Nursery.start()` to wrap them in an
    ExceptionGroup when using `run(..., strict_exception_groups=True)`.

    Regression tests for #2844, where #2611 was initially fixed in a way that
    had unintended side effects.
    """
    raiser_exc: ValueError | ExceptionGroup[ValueError]
    if raise_custom_exc_grp:
        raiser_exc = ExceptionGroup('my group', [ValueError()])
    else:
        raiser_exc = ValueError()

    async def raiser(*, task_status: _core.TaskStatus[None]) -> None:
        if raise_after_started:
            task_status.started()
        raise raiser_exc

    async def start_raiser() -> None:
        try:
            async with _core.open_nursery(strict_exception_groups=start_raiser_strict) as nursery:
                await nursery.start(raiser)
        except BaseExceptionGroup as exc_group:
            if start_raiser_strict:
                raise BaseExceptionGroup('start_raiser nursery custom message', exc_group.exceptions) from None
            raise
    with pytest.raises(BaseException) as exc_info:
        _core.run(start_raiser, strict_exception_groups=run_strict)
    if start_raiser_strict or (run_strict and start_raiser_strict is None):
        assert isinstance(exc_info.value, BaseExceptionGroup)
        if start_raiser_strict:
            assert exc_info.value.message == 'start_raiser nursery custom message'
        assert len(exc_info.value.exceptions) == 1
        should_be_raiser_exc = exc_info.value.exceptions[0]
    else:
        should_be_raiser_exc = exc_info.value
    if isinstance(raiser_exc, ValueError):
        assert should_be_raiser_exc is raiser_exc
    else:
        assert type(should_be_raiser_exc) == type(raiser_exc)
        assert should_be_raiser_exc.message == raiser_exc.message
        assert should_be_raiser_exc.exceptions == raiser_exc.exceptions