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
def _save_checkpoint(self, *args, **kwargs):
    """Saves the checkpoint and exit program."""
    distribute_lib.distribution_strategy_input_api_counter.get_cell(self._platform_device.name, 'PreemptionCheckpointHandler Saving Checkpoint').increase_by(1)
    logging.info('PreemptionCheckpointHandler: Starting saving a checkpoint.')
    if self._platform_device != failure_handling_util.PlatformDevice.INTERNAL_TPU:
        self._checkpointed_runs.assign(self.total_run_calls)
    start_time = time.monotonic()
    with checkpoint_context.preemption_save_context():
        if self._save_fn:
            self._save_fn(*args, **kwargs)
        else:
            self._write_checkpoint_manager.save(*args, **kwargs)
    end_time = time.monotonic()
    logging.info('Checkpoint finished at path %s', self._write_checkpoint_manager.directory)
    self._checkpoint_time = end_time - start_time