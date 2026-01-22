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
def _process_closure(self, closure):
    """Runs a closure with preemption handling."""
    try:
        with self.failure_handler.wait_on_failure(on_failure_fn=lambda e: self._on_closure_failure(closure, e), on_transient_failure_fn=lambda: self._cluster.closure_queue.put_back(closure), on_recovery_fn=self._on_worker_recovery, worker_device_name=self.device_name):
            closure.execute_on(self)
            with metric_utils.monitored_timer('remote_value_fetch'):
                closure.maybe_call_with_output_remote_value(lambda r: r.get())
            self._cluster.closure_queue.mark_finished()
    except Exception as e:
        if not isinstance(e, errors.CancelledError):
            logging.error(' /job:worker/task:%d encountered the following error when processing closure: %r:%s', self.worker_index, e, e)
        closure.maybe_call_with_output_remote_value(lambda r: r._set_error(e))
        self._cluster.closure_queue.mark_failed(e)