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
def _init_init_op(self, init_op=USE_DEFAULT, init_feed_dict=None):
    """Initializes init_op.

    Args:
      init_op: `Operation` to initialize the variables. If set to USE_DEFAULT,
        create an op that initializes all variables and tables.
      init_feed_dict: A dictionary that maps `Tensor` objects to feed values.
        This feed dictionary will be used when `init_op` is evaluated.
    """
    if init_op is Supervisor.USE_DEFAULT:
        init_op = self._get_first_op_from_collection(ops.GraphKeys.INIT_OP)
        if init_op is None:
            init_op = variables.global_variables_initializer()
            ops.add_to_collection(ops.GraphKeys.INIT_OP, init_op)
    self._init_op = init_op
    self._init_feed_dict = init_feed_dict