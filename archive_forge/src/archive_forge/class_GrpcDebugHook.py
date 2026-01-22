from tensorflow.core.protobuf import config_pb2
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.debug.wrappers import dumping_wrapper
from tensorflow.python.debug.wrappers import framework
from tensorflow.python.debug.wrappers import grpc_wrapper
from tensorflow.python.debug.wrappers import local_cli_wrapper
from tensorflow.python.training import session_run_hook
class GrpcDebugHook(session_run_hook.SessionRunHook):
    """A hook that streams debugger-related events to any grpc_debug_server.

  For example, the debugger data server is a grpc_debug_server. The debugger
  data server writes debugger-related events it receives via GRPC to logdir.
  This enables debugging features in Tensorboard such as health pills.

  When the arguments of debug_utils.watch_graph changes, strongly consider
  changing arguments here too so that features are available to tflearn users.

  Can be used as a hook for `tf.compat.v1.train.MonitoredSession`s and
  `tf.estimator.Estimator`s.
  """

    def __init__(self, grpc_debug_server_addresses, watch_fn=None, thread_name_filter=None):
        """Constructs a GrpcDebugHook.

    Args:
      grpc_debug_server_addresses: (`list` of `str`) A list of the gRPC debug
        server addresses, in the format of <host:port>, with or without the
        "grpc://" prefix. For example: ["localhost:7000", "192.168.0.2:8000"]
      watch_fn: A function that allows for customizing which ops to watch at
        which specific steps. See doc of
        `dumping_wrapper.DumpingDebugWrapperSession.__init__` for details.
      thread_name_filter: Regular-expression white list for threads on which the
        wrapper session will be active. See doc of `BaseDebugWrapperSession` for
        more details.
    """
        self._grpc_debug_wrapper_session = None
        self._thread_name_filter = thread_name_filter
        self._grpc_debug_server_addresses = grpc_debug_server_addresses if isinstance(grpc_debug_server_addresses, list) else [grpc_debug_server_addresses]
        self._watch_fn = watch_fn

    def before_run(self, run_context):
        """Called right before a session is run.

    Args:
      run_context: A session_run_hook.SessionRunContext. Encapsulates
        information on the run.

    Returns:
      A session_run_hook.SessionRunArgs object.
    """
        if not self._grpc_debug_wrapper_session:
            self._grpc_debug_wrapper_session = grpc_wrapper.GrpcDebugWrapperSession(run_context.session, self._grpc_debug_server_addresses, watch_fn=self._watch_fn, thread_name_filter=self._thread_name_filter)
        fetches = run_context.original_args.fetches
        feed_dict = run_context.original_args.feed_dict
        watch_options = self._watch_fn(fetches, feed_dict)
        run_options = config_pb2.RunOptions()
        debug_utils.watch_graph(run_options, run_context.session.graph, debug_urls=self._grpc_debug_wrapper_session.prepare_run_debug_urls(fetches, feed_dict), debug_ops=watch_options.debug_ops, node_name_regex_allowlist=watch_options.node_name_regex_allowlist, op_type_regex_allowlist=watch_options.op_type_regex_allowlist, tensor_dtype_regex_allowlist=watch_options.tensor_dtype_regex_allowlist, tolerate_debug_op_creation_failures=watch_options.tolerate_debug_op_creation_failures)
        return session_run_hook.SessionRunArgs(None, feed_dict=None, options=run_options)