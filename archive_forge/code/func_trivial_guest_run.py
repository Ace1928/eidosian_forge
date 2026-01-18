from __future__ import annotations
import asyncio
import contextlib
import contextvars
import queue
import signal
import socket
import sys
import threading
import time
import traceback
import warnings
from functools import partial
from math import inf
from typing import (
import pytest
from outcome import Outcome
import trio
import trio.testing
from trio.abc import Instrument
from ..._util import signal_raise
from .tutil import gc_collect_harder, restore_unraisablehook
def trivial_guest_run(trio_fn: Callable[..., Awaitable[T]], *, in_host_after_start: Callable[[], None] | None=None, **start_guest_run_kwargs: Any) -> T:
    todo: queue.Queue[tuple[str, Outcome[T] | Callable[..., object]]] = queue.Queue()
    host_thread = threading.current_thread()

    def run_sync_soon_threadsafe(fn: Callable[[], object]) -> None:
        nonlocal todo
        if host_thread is threading.current_thread():
            crash = partial(pytest.fail, 'run_sync_soon_threadsafe called from host thread')
            todo.put(('run', crash))
        todo.put(('run', fn))

    def run_sync_soon_not_threadsafe(fn: Callable[[], object]) -> None:
        nonlocal todo
        if host_thread is not threading.current_thread():
            crash = partial(pytest.fail, 'run_sync_soon_not_threadsafe called from worker thread')
            todo.put(('run', crash))
        todo.put(('run', fn))

    def done_callback(outcome: Outcome[T]) -> None:
        nonlocal todo
        todo.put(('unwrap', outcome))
    trio.lowlevel.start_guest_run(trio_fn, run_sync_soon_not_threadsafe, run_sync_soon_threadsafe=run_sync_soon_threadsafe, run_sync_soon_not_threadsafe=run_sync_soon_not_threadsafe, done_callback=done_callback, **start_guest_run_kwargs)
    if in_host_after_start is not None:
        in_host_after_start()
    try:
        while True:
            op, obj = todo.get()
            if op == 'run':
                assert not isinstance(obj, Outcome)
                obj()
            elif op == 'unwrap':
                assert isinstance(obj, Outcome)
                return obj.unwrap()
            else:
                raise NotImplementedError(f'{op!r} not handled')
    finally:
        del todo, run_sync_soon_threadsafe, done_callback