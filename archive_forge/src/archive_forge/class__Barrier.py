import copy
import json
import os
import threading
import time
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.distribute import distribute_coordinator_context
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import coordinator
from tensorflow.python.training import monitored_session
from tensorflow.python.training import server_lib
class _Barrier(object):
    """A reusable barrier class for worker synchronization."""

    def __init__(self, num_participants):
        """Initializes the barrier object.

    Args:
      num_participants: an integer which is the expected number of calls of
        `wait` pass to through this barrier.
    """
        self._num_participants = num_participants
        self._counter = 0
        self._flag = False
        self._local_sense = threading.local()
        self._lock = threading.Lock()
        self._condition = threading.Condition()

    def wait(self):
        """Waits until all other callers reach the same wait call."""
        self._local_sense.value = not self._flag
        with self._lock:
            self._counter += 1
            if self._counter == self._num_participants:
                self._counter = 0
                self._flag = self._local_sense.value
        with self._condition:
            while self._flag != self._local_sense.value:
                self._condition.wait()
            self._condition.notify_all()