import abc
import os
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.checkpoint import checkpoint as trackable_util
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.distribute import distribute_coordinator_context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import resources
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner
from tensorflow.python.training import saver as training_saver
from tensorflow.python.training import session_manager as sm
from tensorflow.python.training import session_run_hook
from tensorflow.python.util import function_utils
from tensorflow.python.util.tf_export import tf_export
def _get_session_manager(self):
    """Gets or creates a SessionManager."""
    if self._session_manager:
        return self._session_manager
    self._session_manager = sm.SessionManager(local_init_op=self._scaffold.local_init_op, local_init_feed_dict=self._scaffold.local_init_feed_dict, ready_op=self._scaffold.ready_op, ready_for_local_init_op=self._scaffold.ready_for_local_init_op, graph=ops.get_default_graph())
    return self._session_manager