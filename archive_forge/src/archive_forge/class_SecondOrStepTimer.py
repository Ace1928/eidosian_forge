import os
import time
import numpy as np
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.client import timeline
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.training.summary_io import SummaryWriterCache
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['train.SecondOrStepTimer'])
class SecondOrStepTimer(_HookTimer):
    """Timer that triggers at most once every N seconds or once every N steps.

  This symbol is also exported to v2 in tf.estimator namespace. See
  https://github.com/tensorflow/estimator/blob/master/tensorflow_estimator/python/estimator/hooks/basic_session_run_hooks.py
  """

    def __init__(self, every_secs=None, every_steps=None):
        self.reset()
        self._every_secs = every_secs
        self._every_steps = every_steps
        if self._every_secs is None and self._every_steps is None:
            raise ValueError('Either every_secs or every_steps should be provided.')
        if self._every_secs is not None and self._every_steps is not None:
            raise ValueError('Can not provide both every_secs and every_steps.')
        super(SecondOrStepTimer, self).__init__()

    def reset(self):
        self._last_triggered_step = None
        self._last_triggered_time = None

    def should_trigger_for_step(self, step):
        """Return true if the timer should trigger for the specified step.

    Args:
      step: Training step to trigger on.

    Returns:
      True if the difference between the current time and the time of the last
      trigger exceeds `every_secs`, or if the difference between the current
      step and the last triggered step exceeds `every_steps`. False otherwise.
    """
        if self._last_triggered_step is None:
            return True
        if self._last_triggered_step == step:
            return False
        if self._every_secs is not None:
            if time.time() >= self._last_triggered_time + self._every_secs:
                return True
        if self._every_steps is not None:
            if step >= self._last_triggered_step + self._every_steps:
                return True
        return False

    def update_last_triggered_step(self, step):
        current_time = time.time()
        if self._last_triggered_time is None:
            elapsed_secs = None
            elapsed_steps = None
        else:
            elapsed_secs = current_time - self._last_triggered_time
            elapsed_steps = step - self._last_triggered_step
        self._last_triggered_time = current_time
        self._last_triggered_step = step
        return (elapsed_secs, elapsed_steps)

    def last_triggered_step(self):
        return self._last_triggered_step