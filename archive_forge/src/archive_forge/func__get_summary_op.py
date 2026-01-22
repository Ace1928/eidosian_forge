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
def _get_summary_op(self):
    """Fetches the summary op either from self._summary_op or self._scaffold.

    Returns:
      Returns a list of summary `Tensor`.
    """
    summary_op = None
    if self._summary_op is not None:
        summary_op = self._summary_op
    elif self._scaffold.summary_op is not None:
        summary_op = self._scaffold.summary_op
    if summary_op is None:
        return None
    if not isinstance(summary_op, list):
        return [summary_op]
    return summary_op