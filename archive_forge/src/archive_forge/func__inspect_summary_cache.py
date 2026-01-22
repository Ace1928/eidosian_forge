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
def _inspect_summary_cache(self, cache, replica_id, step_num, output_stream, tensor_trace_order):
    """Generates a print operation to print trace inspection.

    Args:
      cache: Tensor storing the trace results for the step.
      replica_id: Tensor storing the replica id of the running core.
      step_num: Step number.
      output_stream: Where to print the outputs, e.g., file path, or sys.stderr.
      tensor_trace_order: TensorTraceOrder object holding tensorname to id map.

    Returns:
      The Op to flush the cache to file.
    """

    def _inspect_tensor(tensor):
        """Returns the text to be printed for inspection output."""
        if self._parameters.trace_mode == tensor_tracer_flags.TRACE_MODE_NAN_INF:
            return cond.cond(math_ops.greater(tensor, 0.0), lambda: 'has NaNs/Infs!', lambda: 'has no NaNs or Infs.')
        else:
            return tensor
    if not tensor_trace_order.traced_tensors:
        logging.warn('Inspect mode has no tensors in the cache to check.')
        return control_flow_ops.no_op
    if self._parameters.trace_mode == tensor_tracer_flags.TRACE_MODE_NAN_INF:
        step_has_nan_or_inf = math_ops.greater(math_ops.reduce_sum(cache), 0.0)
    else:
        step_has_nan_or_inf = math_ops.reduce_any(gen_math_ops.logical_or(gen_math_ops.is_nan(cache), gen_math_ops.is_inf(cache)))
    step_error_message = cond.cond(step_has_nan_or_inf, lambda: 'NaNs or Infs in the step!', lambda: 'No numerical issues have been found for the step.')
    if self._parameters.collect_summary_per_core:
        stats = ['\n\n', 'core:', replica_id, ',', 'step:', step_num, '-->', step_error_message, 'Printing tensors for mode:%s...' % self._parameters.trace_mode]
    else:
        stats = ['\n\n', 'step:', step_num, '-->', step_error_message, 'Printing tensors for mode:%s...' % self._parameters.trace_mode]
    for tensor_name, cache_idx in sorted(tensor_trace_order.tensorname_to_cache_idx.items(), key=lambda item: item[1]):
        if self._parameters.collect_summary_per_core:
            stats.extend(['\n', 'core:', replica_id, ',', 'step:', step_num, ',', tensor_name, '-->', _inspect_tensor(cache[cache_idx, 0])])
        else:
            stats.extend(['\n', 'step:', step_num, ',', tensor_name, '-->', _inspect_tensor(cache[cache_idx, 0])])
    return logging_ops.print_v2(*stats, summarize=-1, output_stream=output_stream)