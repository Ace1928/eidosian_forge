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
def test_thread_cache_deref() -> None:
    res = [False]

    class del_me:

        def __call__(self) -> int:
            return 42

        def __del__(self) -> None:
            res[0] = True
    q: Queue[Outcome[int]] = Queue()

    def deliver(outcome: Outcome[int]) -> None:
        q.put(outcome)
    start_thread_soon(del_me(), deliver)
    outcome = q.get()
    assert outcome.unwrap() == 42
    gc_collect_harder()
    assert res[0]