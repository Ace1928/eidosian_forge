import time
import numpy as np
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.client import session
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export
class _CountDownTimer:
    """A timer that tracks a duration since creation."""
    __slots__ = ['_start_time_secs', '_duration_secs']

    def __init__(self, duration_secs):
        self._start_time_secs = time.time()
        self._duration_secs = duration_secs

    def secs_remaining(self):
        diff = self._duration_secs - (time.time() - self._start_time_secs)
        return max(0, diff)