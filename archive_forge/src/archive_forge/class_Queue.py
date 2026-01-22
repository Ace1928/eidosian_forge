import importlib
import logging
import os
import pathlib
import random
import sys
import threading
import time
import urllib.parse
from collections import deque
from types import ModuleType
from typing import (
import numpy as np
import ray
from ray._private.utils import _get_pyarrow_version
from ray.data._internal.arrow_ops.transform_pyarrow import unify_schemas
from ray.data.context import WARN_PREFIX, DataContext
class Queue:
    """A thread-safe queue implementation for multiple producers and consumers.

    Provide `release()` to exit producer threads cooperatively for resource release.
    """

    def __init__(self, queue_size: int):
        self._queue = deque()
        self._threads_exit = False
        self._producer_semaphore = threading.Semaphore(queue_size)
        self._consumer_semaphore = threading.Semaphore(0)
        self._mutex = threading.Lock()

    def put(self, item: Any) -> bool:
        """Put an item into the queue.

        Block if necessary until a free slot is available in queue.
        This method is called by producer threads.

        Returns:
            True if the caller thread should exit immediately.
        """
        self._producer_semaphore.acquire()
        with self._mutex:
            if self._threads_exit:
                return True
            else:
                self._queue.append(item)
        self._consumer_semaphore.release()
        return False

    def get(self) -> Any:
        """Remove and return an item from the queue.

        Block if necessary until an item is available in queue.
        This method is called by consumer threads.
        """
        self._consumer_semaphore.acquire()
        with self._mutex:
            next_item = self._queue.popleft()
        self._producer_semaphore.release()
        return next_item

    def release(self, num_threads: int):
        """Release `num_threads` of producers so they would exit cooperatively."""
        with self._mutex:
            self._threads_exit = True
        for _ in range(num_threads):
            self._producer_semaphore.release()

    def qsize(self):
        """Return the size of the queue."""
        with self._mutex:
            return len(self._queue)