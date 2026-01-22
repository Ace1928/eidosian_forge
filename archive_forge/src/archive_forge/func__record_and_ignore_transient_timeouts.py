import collections
import contextlib
import os
import re
import threading
import time
import weakref
from six.moves import queue
from tensorflow.python.distribute.coordinator import coordinator_context
from tensorflow.python.distribute.coordinator import metric_utils
from tensorflow.python.distribute.coordinator import remote_value
from tensorflow.python.distribute.coordinator import utils
from tensorflow.python.distribute.coordinator import values as values_lib
from tensorflow.python.distribute.coordinator import watchdog
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import executor
from tensorflow.python.eager import function as tf_function
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _record_and_ignore_transient_timeouts(self, e):
    """Records observed timeout error and return if it should be ignored."""
    if self._transient_timeouts_threshold <= 0:
        return False
    if not isinstance(e, errors.DeadlineExceededError):
        return False
    with self._transient_timeouts_lock:
        self._transient_timeouts_count += 1
        if self._transient_timeouts_count >= self._transient_timeouts_threshold:
            return False
    return True