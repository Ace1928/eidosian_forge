from __future__ import annotations
import threading
import time
from contextlib import contextmanager
from queue import Queue
from typing import TYPE_CHECKING, Iterator, NoReturn
import pytest
from .. import _thread_cache
from .._thread_cache import ThreadCache, start_thread_soon
from .tutil import gc_collect_harder, slow
def test_raise_in_deliver(capfd: pytest.CaptureFixture[str]) -> None:
    seen_threads = set()

    def track_threads() -> None:
        seen_threads.add(threading.current_thread())

    def deliver(_: object) -> NoReturn:
        done.set()
        raise RuntimeError("don't do this")
    done = threading.Event()
    start_thread_soon(track_threads, deliver)
    done.wait()
    done = threading.Event()
    start_thread_soon(track_threads, lambda _: done.set())
    done.wait()
    assert len(seen_threads) == 1
    err = capfd.readouterr().err
    assert "don't do this" in err
    assert 'delivering result' in err