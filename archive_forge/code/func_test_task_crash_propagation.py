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
def test_task_crash_propagation() -> None:
    looper_record: list[str] = []

    async def looper() -> None:
        try:
            while True:
                await _core.checkpoint()
        except _core.Cancelled:
            print('looper cancelled')
            looper_record.append('cancelled')

    async def crasher() -> NoReturn:
        raise ValueError('argh')

    async def main() -> None:
        async with _core.open_nursery() as nursery:
            nursery.start_soon(looper)
            nursery.start_soon(crasher)
    with RaisesGroup(Matcher(ValueError, '^argh$')):
        _core.run(main)
    assert looper_record == ['cancelled']