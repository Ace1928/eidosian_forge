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
class TaskSemaphore(object):

    def __init__(self, count):
        """A semaphore for the purpose of limiting the number of tasks

        :param count: The size of semaphore
        """
        self._semaphore = threading.Semaphore(count)

    def acquire(self, tag, blocking=True):
        """Acquire the semaphore

        :param tag: A tag identifying what is acquiring the semaphore. Note
            that this is not really needed to directly use this class but is
            needed for API compatibility with the SlidingWindowSemaphore
            implementation.
        :param block: If True, block until it can be acquired. If False,
            do not block and raise an exception if cannot be aquired.

        :returns: A token (can be None) to use when releasing the semaphore
        """
        logger.debug('Acquiring %s', tag)
        if not self._semaphore.acquire(blocking):
            raise NoResourcesAvailable("Cannot acquire tag '%s'" % tag)

    def release(self, tag, acquire_token):
        """Release the semaphore

        :param tag: A tag identifying what is releasing the semaphore
        :param acquire_token:  The token returned from when the semaphore was
            acquired. Note that this is not really needed to directly use this
            class but is needed for API compatibility with the
            SlidingWindowSemaphore implementation.
        """
        logger.debug('Releasing acquire %s/%s' % (tag, acquire_token))
        self._semaphore.release()