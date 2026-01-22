import collections
import fractions
import functools
import heapq
import inspect
import logging
import math
import random
import threading
from concurrent import futures
import futurist
from futurist import _utils as utils
def _process_immediates():
    with self._waiter:
        try:
            index = self._immediates.popleft()
        except IndexError:
            pass
        else:
            work = self._works[index]
            submitted_at = self._now_func()
            self._log.debug("Submitting immediate callback '%s'", work.name)
            try:
                fut = executor.submit(runner.run, work)
            except _SCHEDULE_RETRY_EXCEPTIONS as exc:
                self._log.error("Failed to submit immediate callback '%s', retrying. Error: %s", work.name, exc)
                self._immediates.append(index)
            else:
                barrier.incr()
                fut.add_done_callback(functools.partial(_on_done, IMMEDIATE, work, index, submitted_at))