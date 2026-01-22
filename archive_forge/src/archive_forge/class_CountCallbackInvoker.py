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
class CountCallbackInvoker(object):
    """An abstraction to invoke a callback when a shared count reaches zero

    :param callback: Callback invoke when finalized count reaches zero
    """

    def __init__(self, callback):
        self._lock = threading.Lock()
        self._callback = callback
        self._count = 0
        self._is_finalized = False

    @property
    def current_count(self):
        with self._lock:
            return self._count

    def increment(self):
        """Increment the count by one"""
        with self._lock:
            if self._is_finalized:
                raise RuntimeError('Counter has been finalized it can no longer be incremented.')
            self._count += 1

    def decrement(self):
        """Decrement the count by one"""
        with self._lock:
            if self._count == 0:
                raise RuntimeError('Counter is at zero. It cannot dip below zero')
            self._count -= 1
            if self._is_finalized and self._count == 0:
                self._callback()

    def finalize(self):
        """Finalize the counter

        Once finalized, the counter never be incremented and the callback
        can be invoked once the count reaches zero
        """
        with self._lock:
            self._is_finalized = True
            if self._count == 0:
                self._callback()