from concurrent import futures
from collections import namedtuple
import copy
import logging
import sys
import threading
from s3transfer.compat import MAXINT
from s3transfer.compat import six
from s3transfer.exceptions import CancelledError, TransferNotDoneError
from s3transfer.utils import FunctionContainer
from s3transfer.utils import TaskSemaphore
def add_failure_cleanup(self, function, *args, **kwargs):
    """Adds a callback to call upon failure"""
    with self._failure_cleanups_lock:
        self._failure_cleanups.append(FunctionContainer(function, *args, **kwargs))