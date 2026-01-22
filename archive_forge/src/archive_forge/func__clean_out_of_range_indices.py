from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sets
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
def _clean_out_of_range_indices(labels, num_classes):
    """Replaces large out-of-range labels by small out-of-range labels.

  Replaces any value in `labels` that is greater or equal to `num_classes` by
  -1. Do this conditionally for efficiency in case there are no such values.

  Args:
    labels: `int64` `Tensor` or `SparseTensor`.
    num_classes: `int64` scalar `Tensor`.
  Returns:
    An `int64` `Tensor` or `SparseTensor` as `labels` with indices greater
    or equal to num_classes replaced by -1.
  """

    def _labels_is_sparse():
        """Returns true is `labels` is a sparse tensor."""
        return isinstance(labels, (sparse_tensor.SparseTensor, sparse_tensor.SparseTensorValue))

    def _clean_out_of_range(values):
        """Replaces by -1 any large out-of-range `values`."""
        return array_ops.where_v2(math_ops.greater_equal(values, num_classes), -1 * array_ops.ones_like(values), values)

    def _clean_labels_out_of_range():
        """Replaces by -1 ane large out-of-range values in `labels`."""
        if _labels_is_sparse():
            return type(labels)(indices=labels.indices, values=_clean_out_of_range(labels.values), dense_shape=labels.dense_shape)
        else:
            return _clean_out_of_range(labels)
    max_labels = math_ops.reduce_max(labels.values if _labels_is_sparse() else labels)
    return cond.cond(math_ops.greater_equal(max_labels, num_classes), _clean_labels_out_of_range, lambda: labels)