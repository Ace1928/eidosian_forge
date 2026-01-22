import random
import time
import functools
import math
import os
import socket
import stat
import string
import logging
import threading
import io
from collections import defaultdict
from botocore.exceptions import IncompleteReadError
from botocore.exceptions import ReadTimeoutError
from s3transfer.compat import SOCKET_ERROR
from s3transfer.compat import rename_file
from s3transfer.compat import seekable
from s3transfer.compat import fallocate
class SlidingWindowSemaphore(TaskSemaphore):
    """A semaphore used to coordinate sequential resource access.

    This class is similar to the stdlib BoundedSemaphore:

    * It's initialized with a count.
    * Each call to ``acquire()`` decrements the counter.
    * If the count is at zero, then ``acquire()`` will either block until the
      count increases, or if ``blocking=False``, then it will raise
      a NoResourcesAvailable exception indicating that it failed to acquire the
      semaphore.

    The main difference is that this semaphore is used to limit
    access to a resource that requires sequential access.  For example,
    if I want to access resource R that has 20 subresources R_0 - R_19,
    this semaphore can also enforce that you only have a max range of
    10 at any given point in time.  You must also specify a tag name
    when you acquire the semaphore.  The sliding window semantics apply
    on a per tag basis.  The internal count will only be incremented
    when the minimum sequence number for a tag is released.

    """

    def __init__(self, count):
        self._count = count
        self._tag_sequences = defaultdict(int)
        self._lowest_sequence = {}
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._pending_release = {}

    def current_count(self):
        with self._lock:
            return self._count

    def acquire(self, tag, blocking=True):
        logger.debug('Acquiring %s', tag)
        self._condition.acquire()
        try:
            if self._count == 0:
                if not blocking:
                    raise NoResourcesAvailable("Cannot acquire tag '%s'" % tag)
                else:
                    while self._count == 0:
                        self._condition.wait()
            sequence_number = self._tag_sequences[tag]
            if sequence_number == 0:
                self._lowest_sequence[tag] = sequence_number
            self._tag_sequences[tag] += 1
            self._count -= 1
            return sequence_number
        finally:
            self._condition.release()

    def release(self, tag, acquire_token):
        sequence_number = acquire_token
        logger.debug('Releasing acquire %s/%s', tag, sequence_number)
        self._condition.acquire()
        try:
            if tag not in self._tag_sequences:
                raise ValueError('Attempted to release unknown tag: %s' % tag)
            max_sequence = self._tag_sequences[tag]
            if self._lowest_sequence[tag] == sequence_number:
                self._lowest_sequence[tag] += 1
                self._count += 1
                self._condition.notify()
                queued = self._pending_release.get(tag, [])
                while queued:
                    if self._lowest_sequence[tag] == queued[-1]:
                        queued.pop()
                        self._lowest_sequence[tag] += 1
                        self._count += 1
                    else:
                        break
            elif self._lowest_sequence[tag] < sequence_number < max_sequence:
                self._pending_release.setdefault(tag, []).append(sequence_number)
                self._pending_release[tag].sort(reverse=True)
            else:
                raise ValueError('Attempted to release unknown sequence number %s for tag: %s' % (sequence_number, tag))
        finally:
            self._condition.release()