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
class _CoordinatedSessionCreator(SessionCreator):
    """Factory for _CoordinatedSession."""

    def __init__(self, session_creator, hooks, stop_grace_period_secs):
        self._session_creator = session_creator
        self._hooks = hooks
        self.coord = None
        self.tf_sess = None
        self._stop_grace_period_secs = stop_grace_period_secs

    def create_session(self):
        """Creates a coordinated session."""
        self.tf_sess = self._session_creator.create_session()
        self.coord = coordinator.Coordinator(clean_stop_exception_types=[])
        if ops.get_collection(ops.GraphKeys.QUEUE_RUNNERS):
            queue_runner.start_queue_runners(sess=self.tf_sess, coord=self.coord)
        for hook in self._hooks:
            hook.after_create_session(self.tf_sess, self.coord)
        return _CoordinatedSession(_HookedSession(self.tf_sess, self._hooks), self.coord, self._stop_grace_period_secs)