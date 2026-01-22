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
def _streaming_sparse_average_precision_at_top_k(labels, predictions_idx, weights=None, metrics_collections=None, updates_collections=None, name=None):
    """Computes average precision@k of predictions with respect to sparse labels.

  `sparse_average_precision_at_top_k` creates two local variables,
  `average_precision_at_<k>/total` and `average_precision_at_<k>/max`, that
  are used to compute the frequency. This frequency is ultimately returned as
  `average_precision_at_<k>`: an idempotent operation that simply divides
  `average_precision_at_<k>/total` by `average_precision_at_<k>/max`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `precision_at_<k>`. Set operations applied to `top_k` and `labels` calculate
  the true positives and false positives weighted by `weights`. Then `update_op`
  increments `true_positive_at_<k>` and `false_positive_at_<k>` using these
  values.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels] or [D1, ... DN], where the latter implies
      num_labels=1. N >= 1 and num_labels is the number of target classes for
      the associated prediction. Commonly, N=1 and `labels` has shape
      [batch_size, num_labels]. [D1, ... DN] must match `predictions_idx`.
      Values should be non-negative. Negative values are ignored.
    predictions_idx: Integer `Tensor` with shape [D1, ... DN, k] where N >= 1.
      Commonly, N=1 and `predictions_idx` has shape [batch size, k]. The final
      dimension contains the top `k` predicted class indices. [D1, ... DN] must
      match `labels`. Values should be in range [0, num_classes).
    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of
      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all
      dimensions must be either `1`, or the same as the corresponding `labels`
      dimension).
    metrics_collections: An optional list of collections that values should
      be added to.
    updates_collections: An optional list of collections that updates should
      be added to.
    name: Name of new update operation, and namespace for other dependent ops.

  Returns:
    mean_average_precision: Scalar `float64` `Tensor` with the mean average
      precision values.
    update: `Operation` that increments variables appropriately, and whose
      value matches `metric`.
  """
    with ops.name_scope(name, 'average_precision_at_top_k', (predictions_idx, labels, weights)) as scope:
        average_precision = _sparse_average_precision_at_top_k(predictions_idx=predictions_idx, labels=labels)
        if weights is not None:
            weights = weights_broadcast_ops.broadcast_weights(math_ops.cast(weights, dtypes.float64), average_precision)
            average_precision = math_ops.multiply(average_precision, weights)
        with ops.name_scope(None, 'max', (average_precision,)) as max_scope:
            max_var = metric_variable([], dtypes.float64, name=max_scope)
            if weights is None:
                batch_max = math_ops.cast(array_ops.size(average_precision, name='batch_max'), dtypes.float64)
            else:
                batch_max = math_ops.reduce_sum(weights, name='batch_max')
            max_update = state_ops.assign_add(max_var, batch_max, name='update')
        with ops.name_scope(None, 'total', (average_precision,)) as total_scope:
            total_var = metric_variable([], dtypes.float64, name=total_scope)
            batch_total = math_ops.reduce_sum(average_precision, name='batch_total')
            total_update = state_ops.assign_add(total_var, batch_total, name='update')

        def precision_across_replicas(_, total_var, max_var):
            return _safe_scalar_div(total_var, max_var, name='mean')
        mean_average_precision = _aggregate_across_replicas(metrics_collections, precision_across_replicas, total_var, max_var)
        update = _safe_scalar_div(total_update, max_update, name=scope)
        if updates_collections:
            ops.add_to_collections(updates_collections, update)
        return (mean_average_precision, update)