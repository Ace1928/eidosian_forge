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
def _skip_op(self, op_id, op, ops_in_exec_path, report_handler):
    """Returns True if we should not trace Op.

    Args:
      op_id: Topological index of the op.
      op: tf.Operation
      ops_in_exec_path: Set of operations that are in the execution path.
      report_handler: An instance of tensor_tracer_report.TTReportHandle.
    Returns:
      True if the op should not be traced, false otherwise.
    """
    if TensorTracer.while_loop_op(op):
        report_handler.instrument_op(op, TensorTracer.reason(op_id, _REASON_WHILELOOP_OP))
        return True
    if TensorTracer.control_flow_op(op):
        report_handler.instrument_op(op, TensorTracer.reason(op_id, _REASON_CONTROLFLOW_OP))
        return True
    if TensorTracer.unsafe_op(op):
        report_handler.instrument_op(op, TensorTracer.reason(op_id, _REASON_UNSAFE_OP))
        return True
    if TensorTracer.device_mismatch(self._tt_config.device_type, op):
        report_handler.instrument_op(op, TensorTracer.reason(op_id, _REASON_DEVICE_MISMATCH))
        return True
    if op not in ops_in_exec_path:
        report_handler.instrument_op(op, TensorTracer.reason(op_id, _REASON_NOT_EXECUTED))
        return True
    if self._is_in_control_flow(op) or not self._is_in_outmost_while_loop(op):
        if not self._should_trace_in_control_flow():
            report_handler.instrument_op(op, TensorTracer.reason(op_id, _REASON_IN_CONTROL_FLOW))
            return True
    if self._is_user_included_op(op):
        report_handler.instrument_op(op, TensorTracer.reason(op_id, _REASON_USER_INCLUDED))
        if tensor_tracer_flags.TT_CHECK_FILTER.value:
            logging.info('USER_INCLUDED op %s', op.name)
        return False
    if not self._inside_op_range(op_id):
        report_handler.instrument_op(op, TensorTracer.reason(op_id, _REASON_OUTSIDE_OP_RANGE))
        return True
    if not self._is_interesting_op(op):
        report_handler.instrument_op(op, TensorTracer.reason(op_id, _REASON_LESS_INTERESTING_OP))
        return True
    if self._is_user_excluded_op(op):
        report_handler.instrument_op(op, TensorTracer.reason(op_id, _REASON_USER_EXCLUDED))
        if tensor_tracer_flags.TT_CHECK_FILTER.value:
            logging.info('USER_EXCLUDED op %s', op.name)
        return True
    return False