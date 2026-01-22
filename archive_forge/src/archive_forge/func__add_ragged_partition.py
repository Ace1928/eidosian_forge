import collections
import re
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging
from tensorflow.python.util.tf_export import tf_export
def _add_ragged_partition(values, partition, tensor_dict, row_splits_dtype, validate):
    """Creates a RaggedTensor from a values tensor and a partition tensor.

  Args:
    values: The values tensor for the new RaggedTensor.
    partition: The partition configuration object.  Specifies the key that
      should be used to look up the partition tensor (unless partition is a
      RaggedFeature.UniformRowLength, in which case there is no partition
      tensor).
    tensor_dict: The dictionary mapping keys to tensors.
    row_splits_dtype: The dtype for the partition tensor.
    validate: Whether to validate that the values form a valid RaggedTensor.

  Returns:
    A new RaggedTensor formed from the values and partition tensors.
  """
    if isinstance(partition, RaggedFeature.UniformRowLength):
        if isinstance(values, ragged_tensor.RaggedTensor):
            length = ops.convert_to_tensor(partition.length, dtype=row_splits_dtype)
            return ragged_tensor.RaggedTensor.from_uniform_row_length(values, length, validate=validate)
        else:
            return array_ops.reshape(values, array_ops.concat([[-1, partition.length], array_ops.shape(values)[1:]], axis=0))
    else:
        partition_t = math_ops.cast(tensor_dict[partition.key], row_splits_dtype)
        if isinstance(partition, RaggedFeature.RowSplits):
            return ragged_tensor.RaggedTensor.from_row_splits(values, partition_t, validate=validate)
        elif isinstance(partition, RaggedFeature.RowLengths):
            return ragged_tensor.RaggedTensor.from_row_lengths(values, partition_t, validate=validate)
        elif isinstance(partition, RaggedFeature.RowStarts):
            return ragged_tensor.RaggedTensor.from_row_starts(values, partition_t, validate=validate)
        elif isinstance(partition, RaggedFeature.RowLimits):
            return ragged_tensor.RaggedTensor.from_row_limits(values, partition_t, validate=validate)
        elif isinstance(partition, RaggedFeature.ValueRowIds):
            return ragged_tensor.RaggedTensor.from_value_rowids(values, partition_t, validate=validate)
        raise ValueError(f'Unhandled partition type {partition!r}')