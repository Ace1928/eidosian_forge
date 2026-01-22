import numpy as np
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_ragged_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import segment_id_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util.tf_export import tf_export
@classmethod
def _convert_row_partition(cls, partition, name, dtype=None, dtype_hint=None):
    """Converts `partition` to Tensors.

    Args:
      partition: A row-partitioning tensor for the `RowPartition` being
        constructed.  I.e., one of: row_splits, row_lengths, row_starts,
        row_limits, value_rowids, uniform_row_length.
      name: The name of the row-partitioning tensor.
      dtype: Optional dtype for the RowPartition. If missing, the type
        is inferred from the type of `uniform_row_length`, dtype_hint,
        or tf.int64.
      dtype_hint: Optional dtype for the RowPartition, used when dtype
        is None. In some cases, a caller may not have a dtype in mind when
        converting to a tensor, so dtype_hint can be used as a soft preference.
        If the conversion to `dtype_hint` is not possible, this argument has no
        effect.

    Returns:
      A tensor equivalent to partition.

    Raises:
      ValueError: if dtype is not int32 or int64.
    """
    if dtype_hint is None:
        dtype_hint = dtypes.int64
    if isinstance(partition, np.ndarray) and partition.dtype == np.int32 and (dtype is None):
        partition = ops.convert_to_tensor(partition, name=name)
    else:
        partition = tensor_conversion.convert_to_tensor_v2(partition, dtype_hint=dtype_hint, dtype=dtype, name=name)
    if partition.dtype not in (dtypes.int32, dtypes.int64):
        raise ValueError('%s must have dtype int32 or int64' % name)
    return partition