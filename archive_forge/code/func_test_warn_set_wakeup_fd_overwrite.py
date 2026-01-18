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
def test_warn_set_wakeup_fd_overwrite() -> None:
    assert signal.set_wakeup_fd(-1) == -1

    async def trio_main(in_host: InHost) -> str:
        return 'ok'
    a, b = socket.socketpair()
    with a, b:
        a.setblocking(False)
        signal.set_wakeup_fd(a.fileno())
        try:
            with pytest.warns(RuntimeWarning, match='signal handling code.*collided'):
                assert trivial_guest_run(trio_main) == 'ok'
        finally:
            assert signal.set_wakeup_fd(-1) == a.fileno()
        signal.set_wakeup_fd(a.fileno())
        try:
            with pytest.warns(RuntimeWarning, match='signal handling code.*collided'):
                assert trivial_guest_run(trio_main, host_uses_signal_set_wakeup_fd=False) == 'ok'
        finally:
            assert signal.set_wakeup_fd(-1) == a.fileno()
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert trivial_guest_run(trio_main) == 'ok'
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert trivial_guest_run(trio_main, host_uses_signal_set_wakeup_fd=True) == 'ok'
        signal.set_wakeup_fd(a.fileno())
        try:

            async def trio_check_wakeup_fd_unaltered(in_host: InHost) -> str:
                fd = signal.set_wakeup_fd(-1)
                assert fd == a.fileno()
                signal.set_wakeup_fd(fd)
                return 'ok'
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                assert trivial_guest_run(trio_check_wakeup_fd_unaltered, host_uses_signal_set_wakeup_fd=True) == 'ok'
        finally:
            assert signal.set_wakeup_fd(-1) == a.fileno()