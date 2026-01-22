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
def add_deadline(self, entry: DeadlineEntry) -> None:
    heapq.heappush(self._deadlines, entry)