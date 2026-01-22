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
def _process_resource_closure(self, closure):
    """Run the given resource closure with preemption handling."""
    assert closure.tag == self.worker_index
    try:
        with self.failure_handler.wait_on_failure(on_failure_fn=self._on_resource_closure_failure, on_transient_failure_fn=lambda: self._process_resource_closure(closure), on_recovery_fn=self._on_worker_recovery, worker_device_name=self.device_name):
            closure.execute_on(self)
    except Exception as e:
        logging.info('[Worker %d] got an exception when processing resource closure', self.worker_index)
        if not isinstance(e, errors.CancelledError):
            logging.error(' /job:worker/task:%d encountered the following error when processing resource closure: %r:%s', self.worker_index, e, e)
        closure.maybe_call_with_output_remote_value(lambda r: r._set_error(e))