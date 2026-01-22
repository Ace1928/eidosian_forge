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
def _preemption_handler(self):
    """A loop that handles preemption.

    This loop waits for signal of worker preemption and upon worker preemption,
    it waits until all workers are back and updates the cluster about the
    restarted workers.
    """
    assert self._should_preemption_thread_run
    while True:
        self._cluster_due_for_update_or_finish.wait()
        if not self._should_preemption_thread_run:
            logging.info('Stopping the failure handing thread.')
            break
        with self._cluster_update_lock:
            try:
                logging.info('Cluster now being recovered.')
                with metric_utils.monitored_timer('server_def_update'):
                    context.context().update_server_def(self._server_def)
                logging.info('Cluster successfully recovered.')
                self._worker_up_cond.notify_all()
                if self._should_preemption_thread_run:
                    self._cluster_due_for_update_or_finish.clear()
            except Exception as e:
                logging.info('Error occurred while updating server def: %s', e)
                try:
                    self._validate_preemption_failure(e)
                except Exception as ps_e:
                    logging.info('Error that occurred while updating server def is not a worker failure. So set it as _error_from_recovery')
                    self._error_from_recovery = ps_e
                    self._worker_up_cond.notify_all()
                    if self._should_preemption_thread_run:
                        self._cluster_due_for_update_or_finish.clear()
                logging.error('Cluster update failed with error: %s. Retrying...', e)