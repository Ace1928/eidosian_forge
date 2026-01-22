from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import tensorflow as tf
from tensorflow.python.training import training_util
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
class _StopAtCheckpointStepHook(tf.compat.v1.train.SessionRunHook):
    """Hook that requests stop at a specified step based on checkpoint.

  Note: We recommend using 'make_stop_at_checkpoint_step_hook` to get the proper
  hook.
  """

    def __init__(self, model_dir, last_step, wait_after_file_check_secs=30):
        """Initializes a `StopAtCheckpointStepHook`.

    This hook requests stop after a last step has been reached. It checks latest
    checkpoint to verify last step is written on disk or not.

    Args:
      model_dir: Directory to read global step from latest checkpoint.
      last_step: Step after which to stop.
      wait_after_file_check_secs: Reading same file by many workers may create
        I/O issues. To throttle that we will wait given secs after each read of
        the file.

    Raises:
      ValueError: If one of the arguments is invalid.
    """
        if last_step is None:
            raise ValueError('last_step must be specified.')
        if model_dir is None:
            raise ValueError('model_dir must be specified.')
        self._model_dir = model_dir
        self._last_step = last_step
        self._wait_after_file_check_secs = wait_after_file_check_secs

    def begin(self):
        self._global_step_tensor = training_util._get_or_create_global_step_read()
        if self._global_step_tensor is None:
            raise RuntimeError('Global step should be created to use StopAtCheckpointStepHook.')

    def before_run(self, run_context):
        return tf.compat.v1.train.SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        global_step = run_values.results + 1
        if global_step >= self._last_step:
            step = estimator_lib._load_global_step_from_checkpoint_dir(self._model_dir)
            if step >= self._last_step:
                run_context.request_stop()
            else:
                time.sleep(self._wait_after_file_check_secs)