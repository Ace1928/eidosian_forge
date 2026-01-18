from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
Generates the RaggedTensor `row_splits` corresponding to a segmentation.

  Returns an integer vector `splits`, where `splits[0] = 0` and
  `splits[i] = splits[i-1] + count(segment_ids==i)`.  Example:

  >>> print(tf.ragged.segment_ids_to_row_splits([0, 0, 0, 2, 2, 3, 4, 4, 4]))
  tf.Tensor([0 3 3 5 6 9], shape=(6,), dtype=int64)

  Args:
    segment_ids: A 1-D integer Tensor.
    num_segments: A scalar integer indicating the number of segments.  Defaults
      to `max(segment_ids) + 1` (or zero if `segment_ids` is empty).
    out_type: The dtype for the return value.  Defaults to `segment_ids.dtype`,
      or `tf.int64` if `segment_ids` does not have a dtype.
    name: A name prefix for the returned tensor (optional).

  Returns:
    A sorted 1-D integer Tensor, with `shape=[num_segments + 1]`.
  