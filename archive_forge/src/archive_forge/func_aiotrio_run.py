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
def aiotrio_run(trio_fn: Callable[..., Awaitable[T]], *, pass_not_threadsafe: bool=True, **start_guest_run_kwargs: Any) -> T:
    loop = asyncio.new_event_loop()

    async def aio_main() -> T:
        trio_done_fut = loop.create_future()

        def trio_done_callback(main_outcome: Outcome[object]) -> None:
            print(f'trio_fn finished: {main_outcome!r}')
            trio_done_fut.set_result(main_outcome)
        if pass_not_threadsafe:
            start_guest_run_kwargs['run_sync_soon_not_threadsafe'] = loop.call_soon
        trio.lowlevel.start_guest_run(trio_fn, run_sync_soon_threadsafe=loop.call_soon_threadsafe, done_callback=trio_done_callback, **start_guest_run_kwargs)
        return (await trio_done_fut).unwrap()
    try:
        return loop.run_until_complete(aio_main())
    finally:
        loop.close()