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
def track_threads() -> None:
    seen_threads.add(threading.current_thread())