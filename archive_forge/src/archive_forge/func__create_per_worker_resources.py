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
def _create_per_worker_resources(self, fn, args=None, kwargs=None):
    """Synchronously create resources on the workers.

    The resources are represented by
    `tf.distribute.experimental.coordinator.RemoteValue`s.

    Args:
      fn: The function to be dispatched to all workers for execution
        asynchronously.
      args: Positional arguments for `fn`.
      kwargs: Keyword arguments for `fn`.

    Returns:
      A `tf.distribute.experimental.coordinator.PerWorkerValues` object, which
      wraps a tuple of `tf.distribute.experimental.coordinator.RemoteValue`
      objects.
    """
    results = []
    for w in self._cluster.workers:
        results.append(w.create_resource(fn, args=args, kwargs=kwargs))
    return PerWorkerValues(tuple(results))