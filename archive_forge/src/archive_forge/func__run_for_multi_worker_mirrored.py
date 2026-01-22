import os
import signal
import sys
import threading
import time
from tensorflow.core.distributed_runtime.preemption import gen_check_preemption_op
from tensorflow.python.checkpoint import checkpoint as checkpoint_lib
from tensorflow.python.checkpoint import checkpoint_context
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute.failure_handling import failure_handling_util
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
def _run_for_multi_worker_mirrored(self, distributed_train_function, *args, **kwargs):
    """PreemptionCheckpointHandler.run implementation for MWMS."""
    try:
        self._check_preemption_and_maybe_checkpoint()
        run_begin_time = time.time()
        result = distributed_train_function(*args, **kwargs)
        new_run_time = time.time() - run_begin_time
        self._run_counter += 1
        self._estimated_run_time = self._estimated_run_time + (new_run_time - self._estimated_run_time) / self._run_counter
    except errors.OpError as e:
        if not self._local_mode:
            logging.info('Propagating error to cluster: %r: %s', e, e)
            try:
                context.context().report_error_to_cluster(e.error_code, e.message)
            except Exception as ex:
                logging.info('Ignoring error during error propagation: %r:%s', ex, ex)
        raise
    return result