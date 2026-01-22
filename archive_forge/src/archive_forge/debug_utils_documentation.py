import re
Add debug tensor watches, denylisting nodes and op types.

  This is similar to `watch_graph()`, but the node names and op types are
  denylisted, instead of allowlisted.

  N.B.:
    1. Under certain circumstances, the `Tensor` may not get actually watched
      (e.g., if the node of the `Tensor` is constant-folded during runtime).
    2. For debugging purposes, the `parallel_iteration` attribute of all
      `tf.while_loop`s in the graph are set to 1 to prevent any node from
      being executed multiple times concurrently. This change does not affect
      subsequent non-debugged runs of the same `tf.while_loop`s.

  Args:
    run_options: An instance of `config_pb2.RunOptions` to be modified.
    graph: An instance of `ops.Graph`.
    debug_ops: (`str` or `list` of `str`) name(s) of the debug op(s) to use. See
      the documentation of `watch_graph` for more details.
    debug_urls: URL(s) to send debug values to, e.g.,
      `file:///tmp/tfdbg_dump_1`, `grpc://localhost:12345`.
    node_name_regex_denylist: Regular-expression denylist for node_name. This
      should be a string, e.g., `"(weight_[0-9]+|bias_.*)"`.
    op_type_regex_denylist: Regular-expression denylist for the op type of
      nodes, e.g., `"(Variable|Add)"`. If both node_name_regex_denylist and
      op_type_regex_denylist are set, the two filtering operations will occur in
      a logical `OR` relation. In other words, a node will be excluded if it
      hits either of the two denylists; a node will be included if and only if
      it hits neither of the denylists.
    tensor_dtype_regex_denylist: Regular-expression denylist for Tensor data
      type, e.g., `"^int.*"`. This denylist operates in logical `OR` relations
      to the two allowlists above.
    tolerate_debug_op_creation_failures: (`bool`) whether debug op creation
      failures (e.g., due to dtype incompatibility) are to be tolerated by not
      throwing exceptions.
    global_step: (`int`) Optional global_step count for this debug tensor watch.
    reset_disk_byte_usage: (`bool`) whether to reset the tracked disk byte
      usage to zero (default: `False`).
  