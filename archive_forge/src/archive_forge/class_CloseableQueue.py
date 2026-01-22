import collections
import os.path
import sys
import threading
import time
from tensorflow.python.client import _pywrap_events_writer
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
class CloseableQueue:
    """Stripped-down fork of the standard library Queue that is closeable."""

    def __init__(self, maxsize=0):
        """Create a queue object with a given maximum size.

    Args:
      maxsize: int size of queue. If <= 0, the queue size is infinite.
    """
        self._maxsize = maxsize
        self._queue = collections.deque()
        self._closed = False
        self._mutex = threading.Lock()
        self._not_empty = threading.Condition(self._mutex)
        self._not_full = threading.Condition(self._mutex)

    def get(self):
        """Remove and return an item from the queue.

    If the queue is empty, blocks until an item is available.

    Returns:
      an item from the queue
    """
        with self._not_empty:
            while not self._queue:
                self._not_empty.wait()
            item = self._queue.popleft()
            self._not_full.notify()
            return item

    def put(self, item):
        """Put an item into the queue.

    If the queue is closed, fails immediately.

    If the queue is full, blocks until space is available or until the queue
    is closed by a call to close(), at which point this call fails.

    Args:
      item: an item to add to the queue

    Raises:
      QueueClosedError: if insertion failed because the queue is closed
    """
        with self._not_full:
            if self._closed:
                raise QueueClosedError()
            if self._maxsize > 0:
                while len(self._queue) == self._maxsize:
                    self._not_full.wait()
                    if self._closed:
                        raise QueueClosedError()
            self._queue.append(item)
            self._not_empty.notify()

    def close(self):
        """Closes the queue, causing any pending or future `put()` calls to fail."""
        with self._not_full:
            self._closed = True
            self._not_full.notify_all()