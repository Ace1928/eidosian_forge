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
class _HookedSession(_WrappedSession):
    """A _WrappedSession that calls hooks during calls to run().

  The list of hooks to call is passed in the constructor.  Before each call
  to `run()` the session calls the `before_run()` method of the hooks, which
  can return additional ops or tensors to run.  These are added to the arguments
  of the call to `run()`.

  When the `run()` call finishes, the session calls the `after_run()` methods of
  the hooks, passing the values returned by the `run()` call corresponding to
  the ops and tensors that each hook requested.

  If any call to the hooks, requests stop via run_context the session will be
  marked as needing to stop and its `should_stop()` method will now return
  `True`.
  """

    def __init__(self, sess, hooks):
        """Initializes a _HookedSession object.

    Args:
      sess: A `tf.compat.v1.Session` or a `_WrappedSession` object.
      hooks: An iterable of `SessionRunHook' objects.
    """
        _WrappedSession.__init__(self, sess)
        self._hooks = hooks
        self._should_stop = False

    def _check_stop(self):
        """See base class."""
        return self._should_stop

    def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
        """See base class."""
        if self.should_stop():
            raise RuntimeError('Run called even after should_stop requested.')
        actual_fetches = {'caller': fetches}
        run_context = session_run_hook.SessionRunContext(original_args=session_run_hook.SessionRunArgs(fetches, feed_dict), session=self._sess)
        options = options or config_pb2.RunOptions()
        feed_dict = self._call_hook_before_run(run_context, actual_fetches, feed_dict, options)
        run_metadata = run_metadata or config_pb2.RunMetadata()
        outputs = _WrappedSession.run(self, fetches=actual_fetches, feed_dict=feed_dict, options=options, run_metadata=run_metadata)
        for hook in self._hooks:
            hook.after_run(run_context, session_run_hook.SessionRunValues(results=outputs[hook] if hook in outputs else None, options=options, run_metadata=run_metadata))
        self._should_stop = self._should_stop or run_context.stop_requested
        return outputs['caller']

    def _call_hook_before_run(self, run_context, fetch_dict, user_feed_dict, options):
        """Calls hooks.before_run and handles requests from hooks."""
        hook_feeds = {}
        for hook in self._hooks:
            request = hook.before_run(run_context)
            if request is not None:
                if request.fetches is not None:
                    fetch_dict[hook] = request.fetches
                if request.feed_dict:
                    self._raise_if_feeds_intersects(hook_feeds, request.feed_dict, 'Same tensor is fed by two hooks.')
                    hook_feeds.update(request.feed_dict)
                if request.options:
                    self._merge_run_options(options, request.options)
        if not hook_feeds:
            return user_feed_dict
        if not user_feed_dict:
            return hook_feeds
        self._raise_if_feeds_intersects(user_feed_dict, hook_feeds, 'Same tensor is fed by a SessionRunHook and user.')
        hook_feeds.update(user_feed_dict)
        return hook_feeds

    def _raise_if_feeds_intersects(self, feeds1, feeds2, message):
        intersection = set(feeds1.keys()) & set(feeds2.keys())
        if intersection:
            raise RuntimeError(message + ' Conflict(s): ' + str(list(intersection)))

    def _merge_run_options(self, options, incoming_options):
        """Merge two instances of RunOptions into the first one.

    During the merger, the numerical fields including trace_level,
    timeout_in_ms, inter_op_thread_pool are set to the larger one of the two.
    The boolean value is set to the logical OR of the two.
    debug_tensor_watch_opts of the original options is extended with that from
    the incoming one.

    Args:
      options: The options to merge into.
      incoming_options: The options to be merged into the first argument.
    """
        options.trace_level = max(options.trace_level, incoming_options.trace_level)
        options.timeout_in_ms = max(options.timeout_in_ms, incoming_options.timeout_in_ms)
        options.inter_op_thread_pool = max(options.inter_op_thread_pool, incoming_options.inter_op_thread_pool)
        options.output_partition_graphs = max(options.output_partition_graphs, incoming_options.output_partition_graphs)
        options.debug_options.debug_tensor_watch_opts.extend(incoming_options.debug_options.debug_tensor_watch_opts)
        options.debug_options.reset_disk_byte_usage = options.debug_options.reset_disk_byte_usage or incoming_options.debug_options.reset_disk_byte_usage
        options.report_tensor_allocations_upon_oom = options.report_tensor_allocations_upon_oom or incoming_options.report_tensor_allocations_upon_oom