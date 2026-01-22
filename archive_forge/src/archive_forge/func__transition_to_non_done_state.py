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
def _transition_to_non_done_state(self, desired_state):
    with self._lock:
        if self.done():
            raise RuntimeError('Unable to transition from done state %s to non-done state %s.' % (self.status, desired_state))
        self._status = desired_state