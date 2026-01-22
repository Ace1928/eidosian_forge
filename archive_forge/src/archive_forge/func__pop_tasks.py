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
def _pop_tasks(self) -> List[Task]:
    tasks = self._tasks
    self._tasks = []
    self._task_set.clear()
    return tasks