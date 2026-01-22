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
def _maybe_expand_labels(labels, predictions):
    """If necessary, expand `labels` along last dimension to match `predictions`.

  Args:
    labels: `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels] or [D1, ... DN]. The latter implies
      num_labels=1, in which case the result is an expanded `labels` with shape
      [D1, ... DN, 1].
    predictions: `Tensor` with shape [D1, ... DN, num_classes].

  Returns:
    `labels` with the same rank as `predictions`.

  Raises:
    ValueError: if `labels` has invalid shape.
  """
    with ops.name_scope(None, 'expand_labels', (labels, predictions)) as scope:
        labels = sparse_tensor.convert_to_tensor_or_sparse_tensor(labels)
        if isinstance(labels, sparse_tensor.SparseTensor):
            return cond.cond(math_ops.equal(array_ops.rank(predictions), array_ops.size(labels.dense_shape) + 1), lambda: sparse_ops.sparse_reshape(labels, shape=array_ops.concat((labels.dense_shape, (1,)), 0), name=scope), lambda: labels)
        labels_rank = labels.get_shape().ndims
        if labels_rank is not None:
            predictions_rank = predictions.get_shape().ndims
            if predictions_rank is not None:
                if predictions_rank == labels_rank:
                    return labels
                if predictions_rank == labels_rank + 1:
                    return array_ops.expand_dims(labels, -1, name=scope)
                raise ValueError(f'Unexpected labels shape {labels.get_shape()} for predictions shape {predictions.get_shape()}. Predictions rank should be the same rank as labels rank or labels rank plus one .')
        return cond.cond(math_ops.equal(array_ops.rank(predictions), array_ops.rank(labels) + 1), lambda: array_ops.expand_dims(labels, -1, name=scope), lambda: labels)