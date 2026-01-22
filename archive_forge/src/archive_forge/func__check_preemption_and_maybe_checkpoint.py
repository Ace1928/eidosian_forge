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
def _check_preemption_and_maybe_checkpoint(self, *args, **kwargs):
    """Checkpoint if any worker has received a preemption signal.

    This function handles preemption signal reported by any worker in the
    cluster. The current implementation relies on the fact that all workers in a
    MultiWorkerMirroredStrategy training cluster have a step number difference
    maximum of 1.
    - If the signal comes from the worker itself (i.e., where this failure
    handler sits), the worker will notify all peers to checkpoint after they
    finish CURRENT_STEP+1 steps, where CURRENT_STEP is the step this worker has
    just finished. And the worker will wait for all peers to acknowledge that
    they have received its preemption signal and the final-step number before
    the worker proceeds on training the final step.
    - If the signal comes from another member in the cluster but NO final-step
    info is available, proceed on training, because it will be available after
    finishing the next step.
    - If the signal comes from some other member in the cluster, and final-step
    info is available, if the worker has not finished these steps yet, keep
    training; otherwise, checkpoint and exit with a cluster-recognized restart
    code.

    Args:
      *args: args for `tf.train.CheckpointManager.save()` to save checkpoint.
      **kwargs: kwargs for `tf.train.CheckpointManager.save()` to save.
    """
    if self._platform_device == failure_handling_util.PlatformDevice.INTERNAL_TPU:
        gen_check_preemption_op.check_preemption(preemption_key=PREEMPTION_KEY)
        return
    if self._final_checkpoint_countdown:
        run_count_config_key = _FINAL_RUN_COUNT_KEY
    else:
        run_count_config_key = _INITIAL_RUN_COUNT_KEY
    if self._received_checkpoint_step.is_set():
        if self._step_to_checkpoint == str(self._run_counter):
            self._save_checkpoint(*args, **kwargs)
            if self._time_to_exit():
                self._stop_poll_termination_signal_thread()
                self._stop_cluster_wise_termination_watcher_thread()
                if self._api_made_checkpoint_manager and (not self._is_chief):
                    gfile.DeleteRecursively(os.path.dirname(self._write_checkpoint_manager.directory))
                logging.info('PreemptionCheckpointHandler: checkpoint saved. Exiting.')
                self._exit_fn()
            else:
                logging.info('Continue training for the grace period.')
                self._final_checkpoint_countdown = True
                self._received_checkpoint_step.clear()
    elif self._received_own_sigterm.is_set():
        if self._final_checkpoint_countdown:
            if self._target_time_for_termination < time.time():
                logging.info('Grace period almost ended. Final call to save a checkpoint!')
            else:
                return
        step_to_save_at = str(self._run_counter + 1)
        logging.info('Termination caught in main thread on preempted worker')
        if self._local_mode:
            self._step_to_checkpoint = step_to_save_at
            self._received_checkpoint_step.set()
        else:
            context.context().set_config_key_value(run_count_config_key, step_to_save_at)
            logging.info('%s set to %s', run_count_config_key, step_to_save_at)
            if not self._local_mode:
                worker_count = multi_worker_util.worker_count(self._cluster_resolver.cluster_spec(), self._cluster_resolver.task_type)
                for i in range(worker_count):
                    context.context().get_config_key_value(f'{_ACKNOWLEDGE_KEY}_{run_count_config_key}_{i}')
                    logging.info('Sigterm acknowledgement from replica %d received', i)
        self._setup_countdown_if_has_grace_period_and_not_already_counting_down()