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
def _get_task_states(self):
    """Get task states and reset to None if coordination service is down."""
    try:
        self._task_states = context.context().get_task_states([('worker', self._num_workers), ('ps', self._num_ps)])
    except (errors.UnavailableError, errors.InternalError) as e:
        if isinstance(e, errors.InternalError) and 'coordination service is not enabled' not in str(e).lower():
            raise
        self._task_states = None
    with self._next_task_state_cond:
        self._next_task_state_cond.notify_all()