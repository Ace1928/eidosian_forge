import collections
import hashlib
import operator
import os
import os.path
import sys
import numpy as np
from tensorflow.core.framework import summary_pb2
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import function
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_case
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import summary_ops_v2 as summary
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import analytics
from tensorflow.python.platform import gfile
from tensorflow.python.platform import remote_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary_iterator
from tensorflow.python.tpu import tensor_tracer_flags
from tensorflow.python.tpu import tensor_tracer_report
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.training import training_util
def _inspect_history_cache(self, cache, replica_id, step_num, tensor_trace_order):
    """Generates a conditional print operation to log differences in tensor values.

    Args:
      cache: Tensor storing the trace results for the step.
      replica_id: Tensor storing the replica id of the running core.
      step_num: Step number.
      tensor_trace_order: TensorTraceOrder object holding tensorname to id map.

    Returns:
      The Op to flush the cache to file.
    """
    if not tensor_trace_order.traced_tensors:
        logging.warn('TT history mode has no tensors in the cache to check.')
        return control_flow_ops.no_op
    stats = ['\n\n', 'core:', replica_id, ',', 'step:', step_num]
    diffs = []
    for tensor_name, cache_idx in sorted(tensor_trace_order.tensorname_to_cache_idx.items(), key=lambda item: item[1]):
        tensor_to_write = cache[cache_idx, 0]
        snapshot_variable = self._create_or_get_tensor_history_values_cache(tensor_to_write.name, tensor_to_write.op.graph, tensor_to_write.shape.as_list(), tensor_to_write.dtype)
        with ops.control_dependencies([snapshot_variable]):
            old_value = state_ops.assign_add(snapshot_variable, 0.0)
        with ops.control_dependencies([old_value]):
            new_value = math_ops.cast(tensor_to_write, dtypes.float32)
            delta = math_ops.abs(math_ops.subtract(old_value, new_value))
            updated = state_ops.assign(snapshot_variable, new_value)
            diffs.append(delta)
        with ops.control_dependencies([updated]):
            new_value_from_var = state_ops.assign_add(snapshot_variable, 0.0)
        stats.extend(['\n', 'core:', replica_id, ',', 'step:', step_num, ',', tensor_name, '-->', old_value, new_value_from_var, delta])
    diff_stack = array_ops_stack.stack(diffs)
    step_max = math_ops.reduce_max(diff_stack)
    return cond.cond(math_ops.greater(step_max, tensor_tracer_flags.DELTA_THRESHOLD.value), lambda: logging_ops.print_v2(*stats, summarize=-1), lambda: control_flow_ops.no_op())