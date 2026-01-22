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
def _close_internal(self, exception_type=None):
    try:
        if not exception_type:
            for h in self._hooks:
                h.end(self._coordinated_creator.tf_sess)
    finally:
        try:
            if self._sess is None:
                raise RuntimeError('Session is already closed.')
            self._sess.close()
        finally:
            self._sess = None
            self._coordinated_creator.tf_sess = None
            self._coordinated_creator.coord = None
            if not self._graph_was_finalized:
                ops.get_default_graph()._unsafe_unfinalize()