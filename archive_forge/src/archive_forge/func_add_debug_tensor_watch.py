import re
def add_debug_tensor_watch(run_options, node_name, output_slot=0, debug_ops='DebugIdentity', debug_urls=None, tolerate_debug_op_creation_failures=False, global_step=-1):
    """Add watch on a `Tensor` to `RunOptions`.

  N.B.:
    1. Under certain circumstances, the `Tensor` may not get actually watched
      (e.g., if the node of the `Tensor` is constant-folded during runtime).
    2. For debugging purposes, the `parallel_iteration` attribute of all
      `tf.while_loop`s in the graph are set to 1 to prevent any node from
      being executed multiple times concurrently. This change does not affect
      subsequent non-debugged runs of the same `tf.while_loop`s.

  Args:
    run_options: An instance of `config_pb2.RunOptions` to be modified.
    node_name: (`str`) name of the node to watch.
    output_slot: (`int`) output slot index of the tensor from the watched node.
    debug_ops: (`str` or `list` of `str`) name(s) of the debug op(s). Can be a
      `list` of `str` or a single `str`. The latter case is equivalent to a
      `list` of `str` with only one element.
      For debug op types with customizable attributes, each debug op string can
      optionally contain a list of attribute names, in the syntax of:
        debug_op_name(attr_name_1=attr_value_1;attr_name_2=attr_value_2;...)
    debug_urls: (`str` or `list` of `str`) URL(s) to send debug values to,
      e.g., `file:///tmp/tfdbg_dump_1`, `grpc://localhost:12345`.
    tolerate_debug_op_creation_failures: (`bool`) Whether to tolerate debug op
      creation failures by not throwing exceptions.
    global_step: (`int`) Optional global_step count for this debug tensor
      watch.
  """
    watch_opts = run_options.debug_options.debug_tensor_watch_opts
    run_options.debug_options.global_step = global_step
    watch = watch_opts.add()
    watch.tolerate_debug_op_creation_failures = tolerate_debug_op_creation_failures
    watch.node_name = node_name
    watch.output_slot = output_slot
    if isinstance(debug_ops, str):
        debug_ops = [debug_ops]
    watch.debug_ops.extend(debug_ops)
    if debug_urls:
        if isinstance(debug_urls, str):
            debug_urls = [debug_urls]
        watch.debug_urls.extend(debug_urls)