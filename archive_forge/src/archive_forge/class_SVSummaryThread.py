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
class SVSummaryThread(coordinator.LooperThread):
    """A thread to save summaries on a timer."""

    def __init__(self, sv, sess):
        """Create a SVSummaryThread.

    Args:
      sv: A `Supervisor`.
      sess: A `Session`.
    """
        super(SVSummaryThread, self).__init__(sv.coord, sv.save_summaries_secs)
        self._sv = sv
        self._sess = sess

    def run_loop(self):
        if self._sv.global_step is not None:
            summary_strs, global_step = self._sess.run([self._sv.summary_op, self._sv.global_step])
        else:
            summary_strs = self._sess.run(self._sv.summary_op)
            global_step = None
        if self._sv.summary_writer:
            logging.info('Recording summary at step %s.', global_step)
            self._sv.summary_writer.add_summary(summary_strs, global_step)