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
def _init_summary_op(self, summary_op=USE_DEFAULT):
    """Initializes summary_op.

    Args:
      summary_op: An Operation that returns a Summary for the event logs. If set
        to USE_DEFAULT, create an op that merges all the summaries.
    """
    if summary_op is Supervisor.USE_DEFAULT:
        summary_op = self._get_first_op_from_collection(ops.GraphKeys.SUMMARY_OP)
        if summary_op is None:
            summary_op = _summary.merge_all()
            if summary_op is not None:
                ops.add_to_collection(ops.GraphKeys.SUMMARY_OP, summary_op)
    self._summary_op = summary_op