import os
import sys
from tensorflow.core.profiler import tfprof_log_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.profiler.internal import flops_registry  # pylint: disable=unused-import
from tensorflow.python.util.tf_export import tf_export
def _get_logged_ops(graph, run_meta=None, add_trace=True, add_trainable_var=True):
    """Extract trainable model parameters and FLOPs for ops from a Graph.

  Args:
    graph: tf.Graph.
    run_meta: RunMetadata proto used to complete shape information.
    add_trace: Whether to add op trace information.
    add_trainable_var: Whether to assign tf.compat.v1.trainable_variables() op
      type '_trainable_variables'.
  Returns:
    logged_ops: dict mapping from op_name to OpLogEntry.
    string_to_id: dict mapping from string to id.
  """
    if run_meta:
        graph = _fill_missing_graph_shape(graph, run_meta)
    op_missing_shape = 0
    logged_ops = {}
    string_to_id = {}
    string_to_id['none'] = len(string_to_id)
    for op in graph.get_operations():
        try:
            stats = ops.get_stats_for_node_def(graph, op.node_def, REGISTERED_FLOP_STATS)
        except ValueError:
            op_missing_shape += 1
            stats = None
        entry = tfprof_log_pb2.OpLogEntry()
        entry.name = op.name
        add_entry = False
        if stats and stats.value:
            entry.float_ops = int(stats.value)
            add_entry = True
        if add_trace:
            if op.traceback:
                for filename, lineno, funcname, line in op.traceback:
                    trace = entry.code_def.traces.add()
                    trace.file_id = _str_id(filename, string_to_id) if filename else 0
                    trace.lineno = lineno if lineno else -1
                    trace.function_id = _str_id(funcname, string_to_id) if funcname else 0
                    trace.line_id = _str_id(line, string_to_id) if line else 0
                    trace.func_start_line = -1
            add_entry = True
        if add_entry:
            logged_ops[entry.name] = entry
    if add_trainable_var:
        for v in graph.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES):
            if v.op.name not in logged_ops:
                entry = tfprof_log_pb2.OpLogEntry()
                entry.name = v.op.name
                entry.types.append(TRAINABLE_VARIABLES)
                logged_ops[entry.name] = entry
            else:
                logged_ops[v.op.name].types.append(TRAINABLE_VARIABLES)
    if op_missing_shape > 0 and (not run_meta):
        sys.stderr.write('%d ops no flops stats due to incomplete shapes.\n' % op_missing_shape)
    return (logged_ops, string_to_id)