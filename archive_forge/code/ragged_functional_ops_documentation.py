from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops.ragged import ragged_config
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
Merges the given list of lists of RowPartitions.

  Args:
    partition_lists: A list of lists of RowPartition.

  Returns:
    A list of RowPartitions, where `result[i]` is formed by merging
    `partition_lists[j][i]` for all `j`, using
    `RowPartition._merge_precomputed_encodings`.
  