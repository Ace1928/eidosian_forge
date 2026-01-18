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
def test_sniffio_integration() -> None:
    with pytest.raises(sniffio.AsyncLibraryNotFoundError):
        sniffio.current_async_library()

    async def check_inside_trio() -> None:
        assert sniffio.current_async_library() == 'trio'

    def check_function_returning_coroutine() -> Awaitable[object]:
        assert sniffio.current_async_library() == 'trio'
        return check_inside_trio()
    _core.run(check_inside_trio)
    with pytest.raises(sniffio.AsyncLibraryNotFoundError):
        sniffio.current_async_library()

    @contextmanager
    def alternate_sniffio_library() -> Generator[None, None, None]:
        prev_token = sniffio.current_async_library_cvar.set('nullio')
        prev_library, sniffio.thread_local.name = (sniffio.thread_local.name, 'nullio')
        try:
            yield
            assert sniffio.current_async_library() == 'nullio'
        finally:
            sniffio.thread_local.name = prev_library
            sniffio.current_async_library_cvar.reset(prev_token)

    async def check_new_task_resets_sniffio_library() -> None:
        with alternate_sniffio_library():
            _core.spawn_system_task(check_inside_trio)
        async with _core.open_nursery() as nursery:
            with alternate_sniffio_library():
                nursery.start_soon(check_inside_trio)
                nursery.start_soon(check_function_returning_coroutine)
    _core.run(check_new_task_resets_sniffio_library)