import enum
import functools
import heapq
import itertools
import signal
import threading
import time
from concurrent.futures import Future
from contextvars import ContextVar
from typing import (
import duet.futuretools as futuretools
def add_ready_callback(self, callback: Callable[['Task'], Any]) -> None:
    self._check_state(TaskState.WAITING)
    self._ready_future.add_done_callback(lambda _: callback(self))