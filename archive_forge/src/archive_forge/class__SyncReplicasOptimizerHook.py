from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import optimizer
from tensorflow.python.training import queue_runner
from tensorflow.python.training import session_manager
from tensorflow.python.training import session_run_hook
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
class _SyncReplicasOptimizerHook(session_run_hook.SessionRunHook):
    """A SessionRunHook handles ops related to SyncReplicasOptimizer."""

    def __init__(self, sync_optimizer, is_chief, num_tokens):
        """Creates hook to handle SyncReplicasOptimizer initialization ops.

    Args:
      sync_optimizer: `SyncReplicasOptimizer` which this hook will initialize.
      is_chief: `Bool`, whether is this a chief replica or not.
      num_tokens: Number of tokens to add to the queue.
    """
        self._sync_optimizer = sync_optimizer
        self._is_chief = is_chief
        self._num_tokens = num_tokens

    def begin(self):
        if self._sync_optimizer._gradients_applied is False:
            raise ValueError('SyncReplicasOptimizer.apply_gradient should be called before using the hook.')
        if self._is_chief:
            self._local_init_op = self._sync_optimizer.chief_init_op
            self._ready_for_local_init_op = self._sync_optimizer.ready_for_local_init_op
            self._q_runner = self._sync_optimizer.get_chief_queue_runner()
            self._init_tokens_op = self._sync_optimizer.get_init_tokens_op(self._num_tokens)
        else:
            self._local_init_op = self._sync_optimizer.local_step_init_op
            self._ready_for_local_init_op = self._sync_optimizer.ready_for_local_init_op
            self._q_runner = None
            self._init_tokens_op = None

    def after_create_session(self, session, coord):
        """Runs SyncReplicasOptimizer initialization ops."""
        local_init_success, msg = session_manager._ready(self._ready_for_local_init_op, session, 'Model is not ready for SyncReplicasOptimizer local init.')
        if not local_init_success:
            raise RuntimeError('Init operations did not make model ready for SyncReplicasOptimizer local_init. Init op: %s, error: %s' % (self._local_init_op.name, msg))
        session.run(self._local_init_op)
        if self._init_tokens_op is not None:
            session.run(self._init_tokens_op)
        if self._q_runner is not None:
            self._q_runner.create_threads(session, coord=coord, daemon=True, start=True)