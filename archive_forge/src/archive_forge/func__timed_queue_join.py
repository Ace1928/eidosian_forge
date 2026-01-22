import os
import threading
from time import sleep, time
from sentry_sdk._queue import Queue, FullError
from sentry_sdk.utils import logger
from sentry_sdk.consts import DEFAULT_QUEUE_SIZE
from sentry_sdk._types import TYPE_CHECKING
def _timed_queue_join(self, timeout):
    deadline = time() + timeout
    queue = self._queue
    queue.all_tasks_done.acquire()
    try:
        while queue.unfinished_tasks:
            delay = deadline - time()
            if delay <= 0:
                return False
            queue.all_tasks_done.wait(timeout=delay)
        return True
    finally:
        queue.all_tasks_done.release()