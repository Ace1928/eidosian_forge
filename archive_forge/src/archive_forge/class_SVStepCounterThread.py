import contextlib
import os
import time
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary as _summary
from tensorflow.python.training import coordinator
from tensorflow.python.training import saver as saver_mod
from tensorflow.python.training import session_manager as session_manager_mod
from tensorflow.python.training import training_util
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
class SVStepCounterThread(coordinator.LooperThread):
    """Threads to count steps and measure their duration."""

    def __init__(self, sv, sess, step_counter=None):
        """Create a `SVStepCounterThread`.

    Args:
      sv: A `Supervisor`.
      sess: A `Session`.
      step_counter: A `Tensor` holding the step counter. By defaults, it uses
        sv.global_step.
    """
        super(SVStepCounterThread, self).__init__(sv.coord, sv.save_summaries_secs)
        self._sv = sv
        self._sess = sess
        self._last_time = 0.0
        self._last_step = 0
        step_counter = sv.global_step if step_counter is None else step_counter
        self._step_counter = step_counter
        self._summary_tag = '%s/sec' % self._step_counter.op.name

    def start_loop(self):
        self._last_time = time.time()
        self._last_step = training_util.global_step(self._sess, self._step_counter)

    def run_loop(self):
        current_step = training_util.global_step(self._sess, self._step_counter)
        added_steps = current_step - self._last_step
        self._last_step = current_step
        current_time = time.time()
        elapsed_time = current_time - self._last_time
        self._last_time = current_time
        if elapsed_time > 0.0:
            steps_per_sec = added_steps / elapsed_time
        else:
            steps_per_sec = float('inf')
        summary = Summary(value=[Summary.Value(tag=self._summary_tag, simple_value=steps_per_sec)])
        if self._sv.summary_writer:
            self._sv.summary_writer.add_summary(summary, current_step)
        logging.log_first_n(logging.INFO, '%s: %g', 10, self._summary_tag, steps_per_sec)