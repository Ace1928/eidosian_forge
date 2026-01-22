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
class async_zip:

    def __init__(self, *largs: it) -> None:
        self.nexts = [obj.__anext__ for obj in largs]

    async def _accumulate(self, f: Callable[[], Awaitable[int]], items: list[int], i: int) -> None:
        items[i] = await f()

    def __aiter__(self) -> async_zip:
        return self

    async def __anext__(self) -> list[int]:
        nexts = self.nexts
        items: list[int] = [-1] * len(nexts)
        try:
            async with _core.open_nursery() as nursery:
                for i, f in enumerate(nexts):
                    nursery.start_soon(self._accumulate, f, items, i)
        except ExceptionGroup as e:
            if len(e.exceptions) == 1 and isinstance(e.exceptions[0], StopAsyncIteration):
                raise e.exceptions[0] from None
            else:
                raise AssertionError('unknown error in _accumulate') from e
        return items