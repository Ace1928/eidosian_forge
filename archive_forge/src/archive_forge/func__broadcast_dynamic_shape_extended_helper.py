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
def _broadcast_dynamic_shape_extended_helper(a: DynamicRaggedShape, b: DynamicRaggedShape) -> Tuple[DynamicRaggedShape, _Broadcaster, _Broadcaster]:
    """Helper for broadcast_dynamic_shape_extended.

  Here, we force:
    a.rank <= b.rank
    2 <= b.rank
    1 <= a.rank
  Args:
    a: a DynamicRaggedShape
    b: a DynamicRaggedShape

  Returns:
    A triple of a shape and two broadcasters.
  """
    assert a.rank <= b.rank
    assert 2 <= b.rank
    assert 1 <= a.rank
    a_rps = a._as_row_partitions()
    b_rps = b._as_row_partitions()
    if len(a_rps) < len(b_rps):
        a_nrows = a[0]
        a_nrows_static = tensor_util.constant_value(a_nrows)
        if a_nrows_static is not None:
            a_nrows = a_nrows_static
        neg_one_a_rp = RowPartition.from_uniform_row_length(uniform_row_length=a_nrows, nrows=1, nvals=a_nrows)
        neg_one_b_rp = b_rps[-(len(a_rps) + 1)]
        neg_one_ac, neg_one_bc = _broadcast_dynamic_shape_first_layer(constant_op.constant(1, dtype=b_rps[0].dtype), neg_one_b_rp.nrows())
        c_zero, ac_zero, bc_zero = _broadcast_dynamic_shape_next_layer(neg_one_ac, neg_one_bc, neg_one_a_rp, neg_one_b_rp)
        b_rps_tail = b_rps[-len(a_rps):] if len(a_rps) >= 1 else []
        c_suffix, ac_layers, bc_layers = _broadcast_dynamic_shape_from_rps(ac_zero, bc_zero, a_rps, b_rps_tail)
        return _broadcast_dynamic_shape_extended_complete(a=a, b=b, b_rps=b_rps, c_suffix=[c_zero] + c_suffix, ac=[ac_zero] + ac_layers, bc_suffix=[neg_one_bc, bc_zero] + bc_layers)
    else:
        assert len(a_rps) == len(b_rps)
        ac_zero, bc_zero = _broadcast_dynamic_shape_first_layer(a_rps[0].nrows(), b_rps[0].nrows())
        c_rps, a_layers, b_layers = _broadcast_dynamic_shape_from_rps(ac_zero, bc_zero, a_rps, b_rps)
        return _broadcast_dynamic_shape_extended_complete(a=a, b=b, b_rps=b_rps, c_suffix=c_rps, ac=[ac_zero] + a_layers, bc_suffix=[bc_zero] + b_layers)