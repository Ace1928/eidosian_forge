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
@tf_contextlib.contextmanager
def watch_preemption_scope(self):
    """Syncs error and maybe save checkpoint for usage with TPUStrategy.

    Note: Usage with `tf.distribute.MultiWorkerMirroredStrategy` does not need
    this API.

    Example usage:

    ```python
    with preemption_checkpoint_handler.watch_preemption_scope():
      while trained_step.numpy() < NUM_STEPS:

        # distributed_train_function contains a call to strategy.run.
        loss += preemption_checkpoint_handler.run(distributed_train_function, args=(next(iterator),))
        trained_step.assign_add(STEPS_PER_TRAIN_FUNCTION)
    ```

    In this workflow, `PreemptionCheckpointHandler.run` will flag preemption
    signal received, and `watch_preemption_scope` will handle the preemption
    signal by saving a checkpoint and then either exit to restart or execute a
    user-passed `exit_fn` in `tf.distribute.experimental.TerminationConfig`. If
    no preemption signal is received during execution of ops and function inside
    the scope, `watch_preemption_scope` ensures the completion of all async op
    and function execution when exiting and will raises exceptions if async
    execution results in an error state.

    Yields:
      None
    """
    if self._platform_device == failure_handling_util.PlatformDevice.INTERNAL_TPU:
        try:
            with context.async_scope():
                yield
        except errors.AbortedError as abort_error:
            if abort_error.experimental_payloads.get(b'type.googleapis.com/tensorflow.distributed_runtime.WorkerPreemption'):
                logging.info('Clearing preemption error to save checkpoint...')
                context.async_clear_error()
                self._save_checkpoint()
                self._exit_fn()
            else:
                raise
    else:
        try:
            yield
        except errors.OpError as e:
            if not self._local_mode:
                logging.info('Propagating error to cluster: %r: %s', e, e)
                try:
                    context.context().report_error_to_cluster(e.error_code, e.message)
                except Exception as ex:
                    logging.info('Ignoring error during error propagation: %r:%s', ex, ex)
            raise