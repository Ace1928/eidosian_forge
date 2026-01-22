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
def _initialize_for_mirrored_and_multi_worker_mirrored(self):
    """Makes configurations and start watchers for MS, MWMS, or OneDevice."""
    if not self._cluster_resolver or not self._cluster_resolver.cluster_spec().jobs:
        self._local_mode = True
        self._id_in_cluster = 'single_worker'
        self._is_chief = True
    else:
        self._local_mode = False
        self._id_in_cluster = str(multi_worker_util.id_in_cluster(self._cluster_resolver.cluster_spec(), self._cluster_resolver.task_type, self._cluster_resolver.task_id))
        self._is_chief = multi_worker_util.is_chief(cluster_spec=self._cluster_resolver.cluster_spec(), task_type=self._cluster_resolver.task_type, task_id=self._cluster_resolver.task_id)
    self._checkpointed_runs = variables.Variable(initial_value=constant_op.constant(0, dtype=dtypes.int64), trainable=False, name=_ITERATION_VARIABLE)
    self._maybe_create_checkpoint_manager()
    if not hasattr(self._write_checkpoint_manager._checkpoint, _ITERATION_VARIABLE):
        setattr(self._write_checkpoint_manager._checkpoint, _ITERATION_VARIABLE, self._checkpointed_runs)
    if not hasattr(self._read_checkpoint_manager._checkpoint, _ITERATION_VARIABLE):
        setattr(self._read_checkpoint_manager._checkpoint, _ITERATION_VARIABLE, self._checkpointed_runs)
    self._read_checkpoint_manager.restore_or_initialize()
    self._final_checkpoint_countdown = False
    self._estimated_run_time = 0
    self._run_counter = self._checkpointed_runs.numpy()
    self._received_own_sigterm = threading.Event()
    self._received_checkpoint_step = threading.Event()
    distribute_lib.distribution_strategy_input_api_counter.get_cell(self._platform_device.name, 'PreemptionCheckpointHandler').increase_by(1)
    if not self._local_mode:
        self._cluster_wise_termination_watcher_thread = threading.Thread(target=self._watch_step_to_save_key, name='PeerTerminationWatcher-%s' % self._id_in_cluster, daemon=True)
        logging.info("Start watcher for peer's signal.")
        self._cluster_wise_termination_watcher_thread.start()
    else:
        self._cluster_wise_termination_watcher_thread = None
    self._poll_termination_signal_thread = None
    if self._termination_watcher_fn:
        self._start_polling_for_termination_signal()
    else:
        self._start_watching_for_signal()