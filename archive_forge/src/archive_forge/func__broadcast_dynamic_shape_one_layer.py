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
def _broadcast_dynamic_shape_one_layer(a, b):
    """Broadcast two vectors, given their shapes.

  Args:
    a: the number of rows in a.
    b: the number of rows in b.

  Returns:
    (layer_a, layer_b, target_shape)
    layer_a is a _LayerBroadcaster from a to the target_shape.
    layer_b is a _LayerBroadcaster from b to the target_shape.
    target_shape is the target_shape

  Raises:
    InvalidArgumentError if the shapes are not consistent.
  """
    a_0 = a[0]
    b_0 = b[0]

    def broadcast_from_a():
        a_layer = array_ops.zeros(b_0, dtype=b_0.dtype)
        b_layer = math_ops.range(b_0)
        target = b
        return [a_layer, b_layer, target]
    a_static = tensor_util.constant_value(a)
    if a_static is not None and a_static[0] == 1:
        [a_gi, b_gi, target] = broadcast_from_a()
        a_layer = _LayerBroadcaster.from_gather_index(a_gi)
        b_layer = _LayerBroadcaster.from_gather_index(b_gi)
        return [a_layer, b_layer, target]

    def broadcast_from_b():
        a_layer = math_ops.range(a_0)
        b_layer = array_ops.zeros(a_0, dtype=a_0.dtype)
        target = a
        return [a_layer, b_layer, target]
    b_static = tensor_util.constant_value(b)
    if b_static is not None and b_static[0] == 1:
        [a_gi, b_gi, target] = broadcast_from_b()
        a_layer = _LayerBroadcaster.from_gather_index(a_gi)
        b_layer = _LayerBroadcaster.from_gather_index(b_gi)
        return [a_layer, b_layer, target]

    def broadcast_noop():
        a_layer = math_ops.range(a_0)
        b_layer = math_ops.range(b_0)
        target = b
        return [a_layer, b_layer, target]
    can_broadcast_from_a = math_ops.equal(a_0, 1)
    can_broadcast_from_b = math_ops.equal(b_0, 1)

    def broadcast_not_from_a():
        return cond.cond(can_broadcast_from_b, true_fn=broadcast_from_b, false_fn=broadcast_noop)
    nrows_equal = math_ops.equal(a_0, b_0)
    can_broadcast = math_ops.logical_or(can_broadcast_from_a, math_ops.logical_or(can_broadcast_from_b, nrows_equal))
    check_can_broadcast = check_ops.assert_equal(can_broadcast, True, message='Cannot broadcast')
    results = cond.cond(can_broadcast_from_a, true_fn=broadcast_from_a, false_fn=broadcast_not_from_a)
    results = [control_flow_ops.with_dependencies([check_can_broadcast], x) for x in results]
    [a_gi, b_gi, target] = results
    a_layer = _LayerBroadcaster.from_gather_index(a_gi)
    b_layer = _LayerBroadcaster.from_gather_index(b_gi)
    return [a_layer, b_layer, target]