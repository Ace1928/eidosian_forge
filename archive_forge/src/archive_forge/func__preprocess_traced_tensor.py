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
def _preprocess_traced_tensor(self, tensor):
    """Computes NAN/Norm/Max on TPUs before sending to CPU.

    Args:
      tensor: The tensor to be traced.
    Returns:
      A tensor that should be input to the trace_function.
    Raises:
      RuntimeError: If the signature is invalid.
    """

    def _detect_nan_inf(tensor):
        """Trace function for detecting any NaN/Inf in the tensor."""
        if tensor.dtype.is_floating:
            mask = math_ops.reduce_any(gen_math_ops.logical_or(gen_math_ops.is_nan(tensor), gen_math_ops.is_inf(tensor)))
            output_tensor = cond.cond(mask, lambda: constant_op.constant([1.0]), lambda: constant_op.constant([0.0]))
        else:
            output_tensor = constant_op.constant([0.0])
        return output_tensor

    def _compute_signature(tensor, tf_op, cast_to_f32=True):
        if cast_to_f32:
            tensor = math_ops.cast(tensor, dtypes.float32)
        output_tensor = tf_op(tensor)
        if not output_tensor.get_shape().is_fully_defined():
            output_tensor = array_ops.reshape(output_tensor, [])
        return output_tensor

    def _show_size(tensor):
        tsize = _compute_signature(tensor, array_ops.size, cast_to_f32=False)
        return math_ops.cast(tsize, dtypes.float32)

    def _show_max(tensor, cast_to_f32=True):
        return _compute_signature(tensor, math_ops.reduce_max, cast_to_f32)

    def _show_min(tensor, cast_to_f32=True):
        return _compute_signature(tensor, math_ops.reduce_min, cast_to_f32)

    def _show_norm(tensor, cast_to_f32=True):
        return _compute_signature(tensor, linalg_ops.norm, cast_to_f32)

    def _show_sparsity(tensor, cast_to_f32=True, tolerance=1e-06):

        def sparsity_fn(tensor):
            non_zeros = math_ops.greater_equal(math_ops.abs(tensor), tolerance)
            nans = math_ops.is_nan(tensor)
            return nn_impl.zero_fraction(math_ops.logical_or(non_zeros, nans))
        return _compute_signature(tensor, sparsity_fn, cast_to_f32)

    def _show_mean_and_variance(tensor, cast_to_f32=True):
        """Returns the mean and variance of the given tensor."""
        if cast_to_f32:
            tensor = math_ops.cast(tensor, dtypes.float32)
        mean, var = nn_impl.moments(array_ops.reshape(tensor, [-1]), axes=[0])
        if not mean.get_shape().is_fully_defined():
            mean = array_ops.reshape(mean, [])
        if not var.get_shape().is_fully_defined():
            var = array_ops.reshape(var, [])
        return (mean, var)

    def _show_max_abs(tensor, cast_to_f32=True):
        return _compute_signature(tensor, lambda t: math_ops.reduce_max(math_ops.abs(t)), cast_to_f32)
    if self._parameters.trace_mode == tensor_tracer_flags.TRACE_MODE_NAN_INF:
        return {self._parameters.trace_mode: _detect_nan_inf(tensor)}
    if self._parameters.trace_mode == tensor_tracer_flags.TRACE_MODE_PART_TENSOR:
        return {self._parameters.trace_mode: tensor}
    if self._parameters.trace_mode in (tensor_tracer_flags.TRACE_MODE_FULL_TENSOR, tensor_tracer_flags.TRACE_MODE_FULL_TENSOR_SUMMARY):
        return {self._parameters.trace_mode: tensor}
    if self._parameters.trace_mode == tensor_tracer_flags.TRACE_MODE_NORM:
        return {self._parameters.trace_mode: array_ops.reshape(_show_norm(tensor), [1])}
    if self._parameters.trace_mode == tensor_tracer_flags.TRACE_MODE_HISTORY:
        return {self._parameters.trace_mode: array_ops.reshape(_show_norm(tensor), [1])}
    if self._parameters.trace_mode == tensor_tracer_flags.TRACE_MODE_MAX_ABS:
        return {self._parameters.trace_mode: _show_max_abs(tensor)}
    if self._parameters.trace_mode == tensor_tracer_flags.TRACE_MODE_SUMMARY:
        tensor = math_ops.cast(tensor, dtypes.float32)
        result_dict = {}
        if _TT_SUMMARY_MEAN in self._signature_types() or _TT_SUMMARY_VAR in self._signature_types():
            mean, variance = _show_mean_and_variance(tensor, cast_to_f32=False)
        for signature_name, _ in sorted(self._signature_types().items(), key=lambda x: x[1]):
            if signature_name == _TT_SUMMARY_NORM:
                signature_result_tensor = _show_norm(tensor, cast_to_f32=False)
            elif signature_name == _TT_SUMMARY_MAX:
                signature_result_tensor = _show_max(tensor, cast_to_f32=False)
            elif signature_name == _TT_SUMMARY_MAX_ABS:
                signature_result_tensor = _show_max_abs(tensor, cast_to_f32=False)
            elif signature_name == _TT_SUMMARY_MIN:
                signature_result_tensor = _show_min(tensor, cast_to_f32=False)
            elif signature_name == _TT_SUMMARY_SPARSITY:
                signature_result_tensor = _show_sparsity(tensor)
            elif signature_name == _TT_SUMMARY_SIZE:
                signature_result_tensor = _show_size(tensor)
            elif signature_name == _TT_SUMMARY_MEAN:
                signature_result_tensor = mean
            elif signature_name == _TT_SUMMARY_VAR:
                signature_result_tensor = variance
            else:
                raise ValueError('Unknown signature type :%s.' % signature_name)
            result_dict[signature_name] = signature_result_tensor
        return result_dict
    raise RuntimeError('Unsupported signature for trace mode %s.' % self._parameters.trace_mode)