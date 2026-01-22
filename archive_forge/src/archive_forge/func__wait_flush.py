import os
import threading
from time import sleep, time
from sentry_sdk._queue import Queue, FullError
from sentry_sdk.utils import logger
from sentry_sdk.consts import DEFAULT_QUEUE_SIZE
from sentry_sdk._types import TYPE_CHECKING
def _wait_flush(self, timeout, callback):
    initial_timeout = min(0.1, timeout)
    if not self._timed_queue_join(initial_timeout):
        pending = self._queue.qsize() + 1
        logger.debug('%d event(s) pending on flush', pending)
        if callback is not None:
            callback(pending, timeout)
        if not self._timed_queue_join(timeout - initial_timeout):
            pending = self._queue.qsize() + 1
            logger.error('flush timed out, dropped %s events', pending)