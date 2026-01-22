from enum import Enum
import functools
import weakref
import numpy as np
from tensorflow.python.compat import compat
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.ops.parallel_for import control_flow_ops as parallel_control_flow_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import tf_decorator
def _update_confusion_matrix_variables_optimized(variables_to_update, y_true, y_pred, thresholds, multi_label=False, sample_weights=None, label_weights=None, thresholds_with_epsilon=False):
    """Update confusion matrix variables with memory efficient alternative.

  Note that the thresholds need to be evenly distributed within the list, eg,
  the diff between consecutive elements are the same.

  To compute TP/FP/TN/FN, we are measuring a binary classifier
    C(t) = (predictions >= t)
  at each threshold 't'. So we have
    TP(t) = sum( C(t) * true_labels )
    FP(t) = sum( C(t) * false_labels )

  But, computing C(t) requires computation for each t. To make it fast,
  observe that C(t) is a cumulative integral, and so if we have
    thresholds = [t_0, ..., t_{n-1}];  t_0 < ... < t_{n-1}
  where n = num_thresholds, and if we can compute the bucket function
    B(i) = Sum( (predictions == t), t_i <= t < t{i+1} )
  then we get
    C(t_i) = sum( B(j), j >= i )
  which is the reversed cumulative sum in tf.cumsum().

  We can compute B(i) efficiently by taking advantage of the fact that
  our thresholds are evenly distributed, in that
    width = 1.0 / (num_thresholds - 1)
    thresholds = [0.0, 1*width, 2*width, 3*width, ..., 1.0]
  Given a prediction value p, we can map it to its bucket by
    bucket_index(p) = floor( p * (num_thresholds - 1) )
  so we can use tf.math.unsorted_segment_sum() to update the buckets in one
  pass.

  Consider following example:
  y_true = [0, 0, 1, 1]
  y_pred = [0.1, 0.5, 0.3, 0.9]
  thresholds = [0.0, 0.5, 1.0]
  num_buckets = 2   # [0.0, 1.0], (1.0, 2.0]
  bucket_index(y_pred) = tf.math.floor(y_pred * num_buckets)
                       = tf.math.floor([0.2, 1.0, 0.6, 1.8])
                       = [0, 0, 0, 1]
  # The meaning of this bucket is that if any of the label is true,
  # then 1 will be added to the corresponding bucket with the index.
  # Eg, if the label for 0.2 is true, then 1 will be added to bucket 0. If the
  # label for 1.8 is true, then 1 will be added to bucket 1.
  #
  # Note the second item "1.0" is floored to 0, since the value need to be
  # strictly larger than the bucket lower bound.
  # In the implementation, we use tf.math.ceil() - 1 to achieve this.
  tp_bucket_value = tf.math.unsorted_segment_sum(true_labels, bucket_indices,
                                                 num_segments=num_thresholds)
                  = [1, 1, 0]
  # For [1, 1, 0] here, it means there is 1 true value contributed by bucket 0,
  # and 1 value contributed by bucket 1. When we aggregate them to together,
  # the result become [a + b + c, b + c, c], since large thresholds will always
  # contribute to the value for smaller thresholds.
  true_positive = tf.math.cumsum(tp_bucket_value, reverse=True)
                = [2, 1, 0]

  This implementation exhibits a run time and space complexity of O(T + N),
  where T is the number of thresholds and N is the size of predictions.
  Metrics that rely on standard implementation instead exhibit a complexity of
  O(T * N).

  Args:
    variables_to_update: Dictionary with 'tp', 'fn', 'tn', 'fp' as valid keys
      and corresponding variables to update as values.
    y_true: A floating point `Tensor` whose shape matches `y_pred`. Will be cast
      to `bool`.
    y_pred: A floating point `Tensor` of arbitrary shape and whose values are in
      the range `[0, 1]`.
    thresholds: A sorted floating point `Tensor` with value in `[0, 1]`.
      It need to be evenly distributed (the diff between each element need to be
      the same).
    multi_label: Optional boolean indicating whether multidimensional
      prediction/labels should be treated as multilabel responses, or flattened
      into a single label. When True, the valus of `variables_to_update` must
      have a second dimension equal to the number of labels in y_true and
      y_pred, and those tensors must not be RaggedTensors.
    sample_weights: Optional `Tensor` whose rank is either 0, or the same rank
      as `y_true`, and must be broadcastable to `y_true` (i.e., all dimensions
      must be either `1`, or the same as the corresponding `y_true` dimension).
    label_weights: Optional tensor of non-negative weights for multilabel
      data. The weights are applied when calculating TP, FP, FN, and TN without
      explicit multilabel handling (i.e. when the data is to be flattened).
    thresholds_with_epsilon: Optional boolean indicating whether the leading and
      tailing thresholds has any epsilon added for floating point imprecisions.
      It will change how we handle the leading and tailing bucket.

  Returns:
    Update op.
  """
    num_thresholds = thresholds.shape.as_list()[0]
    if sample_weights is None:
        sample_weights = 1.0
    else:
        sample_weights = weights_broadcast_ops.broadcast_weights(math_ops.cast(sample_weights, dtype=y_pred.dtype), y_pred)
        if not multi_label:
            sample_weights = array_ops.reshape(sample_weights, [-1])
    if label_weights is None:
        label_weights = 1.0
    else:
        label_weights = array_ops.expand_dims(label_weights, 0)
        label_weights = weights_broadcast_ops.broadcast_weights(label_weights, y_pred)
        if not multi_label:
            label_weights = array_ops.reshape(label_weights, [-1])
    weights = math_ops.multiply(sample_weights, label_weights)
    y_pred = clip_ops.clip_by_value(y_pred, clip_value_min=0.0, clip_value_max=1.0)
    y_true = math_ops.cast(math_ops.cast(y_true, dtypes.bool), y_true.dtype)
    if not multi_label:
        y_true = array_ops.reshape(y_true, [-1])
        y_pred = array_ops.reshape(y_pred, [-1])
    true_labels = math_ops.multiply(y_true, weights)
    false_labels = math_ops.multiply(1.0 - y_true, weights)
    bucket_indices = math_ops.ceil(y_pred * (num_thresholds - 1)) - 1
    if thresholds_with_epsilon:
        bucket_indices = nn_ops.relu(bucket_indices)
    bucket_indices = math_ops.cast(bucket_indices, dtypes.int32)
    if multi_label:
        true_labels = array_ops.transpose_v2(true_labels)
        false_labels = array_ops.transpose_v2(false_labels)
        bucket_indices = array_ops.transpose_v2(bucket_indices)

        def gather_bucket(label_and_bucket_index):
            label, bucket_index = (label_and_bucket_index[0], label_and_bucket_index[1])
            return math_ops.unsorted_segment_sum(data=label, segment_ids=bucket_index, num_segments=num_thresholds)
        tp_bucket_v = parallel_control_flow_ops.vectorized_map(gather_bucket, (true_labels, bucket_indices))
        fp_bucket_v = parallel_control_flow_ops.vectorized_map(gather_bucket, (false_labels, bucket_indices))
        tp = array_ops.transpose_v2(math_ops.cumsum(tp_bucket_v, reverse=True, axis=1))
        fp = array_ops.transpose_v2(math_ops.cumsum(fp_bucket_v, reverse=True, axis=1))
    else:
        tp_bucket_v = math_ops.unsorted_segment_sum(data=true_labels, segment_ids=bucket_indices, num_segments=num_thresholds)
        fp_bucket_v = math_ops.unsorted_segment_sum(data=false_labels, segment_ids=bucket_indices, num_segments=num_thresholds)
        tp = math_ops.cumsum(tp_bucket_v, reverse=True)
        fp = math_ops.cumsum(fp_bucket_v, reverse=True)
    if ConfusionMatrix.TRUE_NEGATIVES in variables_to_update or ConfusionMatrix.FALSE_NEGATIVES in variables_to_update:
        if multi_label:
            total_true_labels = math_ops.reduce_sum(true_labels, axis=1)
            total_false_labels = math_ops.reduce_sum(false_labels, axis=1)
        else:
            total_true_labels = math_ops.reduce_sum(true_labels)
            total_false_labels = math_ops.reduce_sum(false_labels)
    update_ops = []
    if ConfusionMatrix.TRUE_POSITIVES in variables_to_update:
        variable = variables_to_update[ConfusionMatrix.TRUE_POSITIVES]
        update_ops.append(variable.assign_add(tp))
    if ConfusionMatrix.FALSE_POSITIVES in variables_to_update:
        variable = variables_to_update[ConfusionMatrix.FALSE_POSITIVES]
        update_ops.append(variable.assign_add(fp))
    if ConfusionMatrix.TRUE_NEGATIVES in variables_to_update:
        variable = variables_to_update[ConfusionMatrix.TRUE_NEGATIVES]
        tn = total_false_labels - fp
        update_ops.append(variable.assign_add(tn))
    if ConfusionMatrix.FALSE_NEGATIVES in variables_to_update:
        variable = variables_to_update[ConfusionMatrix.FALSE_NEGATIVES]
        fn = total_true_labels - tp
        update_ops.append(variable.assign_add(fn))
    return control_flow_ops.group(update_ops)