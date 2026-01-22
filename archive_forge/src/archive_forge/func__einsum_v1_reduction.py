import collections
import functools
import re
import string
import numpy as np
import opt_einsum
from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import gen_special_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _einsum_v1_reduction(t0, t0_axis_labels, t1, t1_axis_labels, axes_to_sum):
    """Helper for einsum() that computes the result of a two-argument einsum().

  Args:
    t0: a `Tensor`
    t0_axis_labels: a string of axis labels.  This string's length must equal
      the rank of t0.
    t1: a `Tensor`
    t1_axis_labels: a string to axis labels.  This string's length must equal
      the rank of t1.
    axes_to_sum: set of labels of axes to be summed over

  Returns:
    A `Tensor` whose elements are obtained by summing, over all axes in
    `axes_to_sum`, the corresponding elements of `t0` and `t1`.

    For example, if t0_axis_labels == 'abijk', t1_axis_labels == 'acjkl', and
    axes_to_sum == {j,k}, this will return a tensor x where

      out[a,b,c,i,l] = sum_j sum_k t0[a,b,i,j,k] * t1[a,c,j,k,l]

  Raises:
    ValueError: if the rank of `t0` does not match the length of
      `t0_axis_labels`, or that of `t1` does not match the length of
      `t1_axis_labels`.
  """
    if len(t0_axis_labels) != len(t0.shape):
        raise ValueError(f'Tensor `t0` of rank {len(t0.shape)} does not match einsum reduction of length {len(t0_axis_labels)}.')
    if len(t1_axis_labels) != len(t1.shape):
        raise ValueError(f'Tensor `t1` of rank {len(t1.shape)} does not match einsum reduction of length {len(t1_axis_labels)}')
    assert all((a in t0_axis_labels and a in t1_axis_labels for a in axes_to_sum))
    preserved_axes = (set(t0_axis_labels) & set(t1_axis_labels)) - axes_to_sum
    broadcast_axes = {}
    for i, sym_list in enumerate([t0_axis_labels, t1_axis_labels]):
        broadcast_axes[i] = set(sym_list) - preserved_axes - axes_to_sum

    def sort_key(input_index, a):
        if a in preserved_axes:
            return (-1, a)
        elif input_index == 0 and a in broadcast_axes[0] or (input_index == 1 and a in axes_to_sum):
            return (0, a)
        else:
            return (1, a)
    axis_labels = [t0_axis_labels, t1_axis_labels]
    sorted_axes = [sorted(sym_list, key=lambda a: sort_key(i, a)) for i, sym_list in enumerate(axis_labels)]
    inputs = [t0, t1]
    for i, axes_str in enumerate(axis_labels):
        perm = [axes_str.find(a) for a in sorted_axes[i]]
        inputs[i] = _transpose_if_necessary(inputs[i], perm)
    t0, t1 = inputs
    if not axes_to_sum:
        for _ in broadcast_axes[1]:
            t0 = array_ops.expand_dims(t0, -1)
        for _ in broadcast_axes[0]:
            t1 = array_ops.expand_dims(t1, len(preserved_axes))
        product = math_ops.multiply(t0, t1)
        product_axes = sorted_axes[0] + sorted_axes[1][len(preserved_axes):]
        return (product, ''.join(product_axes))
    else:
        t0_shape = _get_shape(t0)
        num_broadcast_elements_t0 = _total_size(t0_shape[len(preserved_axes):-len(axes_to_sum)])
        num_summed_elements = _total_size(t0_shape[-len(axes_to_sum):])
        new_shape = t0_shape[:len(preserved_axes)] + [num_broadcast_elements_t0, num_summed_elements]
        t0 = _reshape_if_necessary(t0, new_shape)
        t1_shape = _get_shape(t1)
        num_broadcast_elements_t1 = _total_size(t1_shape[len(preserved_axes) + len(axes_to_sum):])
        new_shape = t1_shape[:len(preserved_axes)] + [num_summed_elements, num_broadcast_elements_t1]
        t1 = _reshape_if_necessary(t1, new_shape)
        product = math_ops.matmul(t0, t1)
        uncompacted_shape = t0_shape[:len(preserved_axes) + len(broadcast_axes[0])] + t1_shape[len(t1_shape) - len(broadcast_axes[1]):]
        product = _reshape_if_necessary(product, uncompacted_shape)
        product_axes = sorted_axes[0][:len(preserved_axes) + len(broadcast_axes[0])] + sorted_axes[1][len(sorted_axes[1]) - len(broadcast_axes[1]):]
        return (product, ''.join(product_axes))