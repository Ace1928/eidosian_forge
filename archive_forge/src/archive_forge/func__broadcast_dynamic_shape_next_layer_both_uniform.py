import abc
from typing import Any, Iterable, Optional, Sequence, Tuple, Union
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.ops.ragged.row_partition import RowPartitionSpec
from tensorflow.python.types import core
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _broadcast_dynamic_shape_next_layer_both_uniform(ac_0: _LayerBroadcaster, bc_0: _LayerBroadcaster, a_1: RowPartition, b_1: RowPartition) -> Tuple[RowPartition, _LayerBroadcaster, _LayerBroadcaster]:
    """Broadcast target and next layer broadcaster of two uniform dynamic shapes.

     *--ac_0-->*<--bc_0--*
     |         |         |
    a_1       c_1       b_1
     |         |         |
     V         V         V
     *--ac_1-->*<--bc_1--*

  Args:
    ac_0: _LayerBroadcaster from a to c in the previous layer.
    bc_0: _LayerBroadcaster from b to c in the previous layer.
    a_1: a RowPartition for the next layer of a.
    b_1: a RowPartition for the next layer of b.

  Returns:
    (c_1, ac_1, bc_1)
    c_1: a RowPartition for the next layer of the dynamic shape.
    ac_1: _LayerBroadcaster from a to c in the next layer.
    bc_1: _LayerBroadcaster from b to c in the next layer.
  """
    if not isinstance(ac_0, _LayerBroadcaster):
        raise TypeError('ac_0 should be a _LayerBroadcaster')
    if not isinstance(bc_0, _LayerBroadcaster):
        raise TypeError('bc_0 should be a _LayerBroadcaster')
    if not isinstance(a_1, RowPartition):
        raise TypeError('a_1 should be a RowPartition')
    if not isinstance(b_1, RowPartition):
        raise TypeError('b_1 should be a RowPartition')
    assert a_1.is_uniform()
    assert b_1.is_uniform()
    static_a_1 = tensor_util.constant_value(a_1.uniform_row_length())
    static_b_1 = tensor_util.constant_value(b_1.uniform_row_length())
    if static_a_1 is not None:
        if static_a_1 == static_b_1:
            [ac_1, _] = _broadcast_half(ac_0, a_1)
            [bc_1, _] = _broadcast_half(bc_0, b_1)
            c_1 = RowPartition.from_uniform_row_length(static_a_1, nrows=ac_0.dest_nrows())
            return [c_1, ac_1, bc_1]
        elif static_a_1 == 1:
            [bc_1, c_1b] = _broadcast_half(bc_0, b_1)
            ac_1 = _LayerBroadcaster.from_gather_index(array_ops.gather(ac_0.gather_index, c_1b.value_rowids()))
            c_1 = RowPartition.from_uniform_row_length(b_1.uniform_row_length(), nrows=bc_0.dest_nrows())
            return [c_1, ac_1, bc_1]
    if static_b_1 == 1:
        [ac_1, c_1a] = _broadcast_half(ac_0, a_1)
        bc_1 = _LayerBroadcaster.from_gather_index(array_ops.gather(bc_0.gather_index, c_1a.value_rowids()))
        c_1 = RowPartition.from_uniform_row_length(a_1.uniform_row_length(), nrows=ac_0.dest_nrows())
        return [c_1, ac_1, bc_1]

    def broadcast_noop():
        [ac_1, _] = _broadcast_half(ac_0, a_1)
        [bc_1, _] = _broadcast_half(bc_0, b_1)
        return [a_1.uniform_row_length(), ac_1.gather_index, bc_1.gather_index]

    def broadcast_a():
        [bc_1, c_1b] = _broadcast_half(bc_0, b_1)
        ac_1_gather_index = array_ops.gather(ac_0.gather_index, c_1b.value_rowids())
        return [b_1.uniform_row_length(), ac_1_gather_index, bc_1.gather_index]

    def broadcast_b():
        [ac_1, c_1a] = _broadcast_half(ac_0, a_1)
        bc_1_gather_index = array_ops.gather(bc_0.gather_index, c_1a.value_rowids())
        return [a_1.uniform_row_length(), ac_1.gather_index, bc_1_gather_index]
    can_broadcast_b = math_ops.equal(b_1.uniform_row_length(), 1)

    def no_broadcast_a():
        return cond.cond(can_broadcast_b, true_fn=broadcast_b, false_fn=broadcast_noop)
    can_broadcast_a = math_ops.equal(a_1.uniform_row_length(), 1)
    broadcast_asserts = [check_ops.assert_equal(math_ops.logical_or(math_ops.logical_or(can_broadcast_a, can_broadcast_b), math_ops.equal(a_1.uniform_row_length(), b_1.uniform_row_length())), True)]
    result = cond.cond(can_broadcast_a, true_fn=broadcast_a, false_fn=no_broadcast_a)
    [c_1_uniform_row_length, ac_1_gather_index, bc_1_gather_index] = [control_flow_ops.with_dependencies(broadcast_asserts, x) for x in result]
    c_1 = RowPartition.from_uniform_row_length(c_1_uniform_row_length, nvals=c_1_uniform_row_length * ac_0.dest_nrows(), nrows=ac_0.dest_nrows())
    ac_1 = _LayerBroadcaster.from_gather_index(ac_1_gather_index)
    bc_1 = _LayerBroadcaster.from_gather_index(bc_1_gather_index)
    return [c_1, ac_1, bc_1]