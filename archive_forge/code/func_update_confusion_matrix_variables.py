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
def update_confusion_matrix_variables(variables_to_update, y_true, y_pred, thresholds, top_k=None, class_id=None, sample_weight=None, multi_label=False, label_weights=None, thresholds_distributed_evenly=False):
    """Returns op to update the given confusion matrix variables.

  For every pair of values in y_true and y_pred:

  true_positive: y_true == True and y_pred > thresholds
  false_negatives: y_true == True and y_pred <= thresholds
  true_negatives: y_true == False and y_pred <= thresholds
  false_positive: y_true == False and y_pred > thresholds

  The results will be weighted and added together. When multiple thresholds are
  provided, we will repeat the same for every threshold.

  For estimation of these metrics over a stream of data, the function creates an
  `update_op` operation that updates the given variables.

  If `sample_weight` is `None`, weights default to 1.
  Use weights of 0 to mask values.

  Args:
    variables_to_update: Dictionary with 'tp', 'fn', 'tn', 'fp' as valid keys
      and corresponding variables to update as values.
    y_true: A `Tensor` whose shape matches `y_pred`. Will be cast to `bool`.
    y_pred: A floating point `Tensor` of arbitrary shape and whose values are in
      the range `[0, 1]`.
    thresholds: A float value, float tensor, python list, or tuple of float
      thresholds in `[0, 1]`, or NEG_INF (used when top_k is set).
    top_k: Optional int, indicates that the positive labels should be limited to
      the top k predictions.
    class_id: Optional int, limits the prediction and labels to the class
      specified by this argument.
    sample_weight: Optional `Tensor` whose rank is either 0, or the same rank as
      `y_true`, and must be broadcastable to `y_true` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `y_true` dimension).
    multi_label: Optional boolean indicating whether multidimensional
      prediction/labels should be treated as multilabel responses, or flattened
      into a single label. When True, the valus of `variables_to_update` must
      have a second dimension equal to the number of labels in y_true and
      y_pred, and those tensors must not be RaggedTensors.
    label_weights: (optional) tensor of non-negative weights for multilabel
      data. The weights are applied when calculating TP, FP, FN, and TN without
      explicit multilabel handling (i.e. when the data is to be flattened).
    thresholds_distributed_evenly: Boolean, whether the thresholds are evenly
      distributed within the list. An optimized method will be used if this is
      the case. See _update_confusion_matrix_variables_optimized() for more
      details.

  Returns:
    Update op.

  Raises:
    ValueError: If `y_pred` and `y_true` have mismatched shapes, or if
      `sample_weight` is not `None` and its shape doesn't match `y_pred`, or if
      `variables_to_update` contains invalid keys.
  """
    if multi_label and label_weights is not None:
        raise ValueError('`label_weights` for multilabel data should be handled outside of `update_confusion_matrix_variables` when `multi_label` is True.')
    if variables_to_update is None:
        return
    if not any((key for key in variables_to_update if key in list(ConfusionMatrix))):
        raise ValueError('Please provide at least one valid confusion matrix variable to update. Valid variable key options are: "{}". Received: "{}"'.format(list(ConfusionMatrix), variables_to_update.keys()))
    variable_dtype = list(variables_to_update.values())[0].dtype
    y_true = math_ops.cast(y_true, dtype=variable_dtype)
    y_pred = math_ops.cast(y_pred, dtype=variable_dtype)
    if thresholds_distributed_evenly:
        thresholds_with_epsilon = thresholds[0] < 0.0 or thresholds[-1] > 1.0
    thresholds = tensor_conversion.convert_to_tensor_v2_with_dispatch(thresholds, dtype=variable_dtype)
    num_thresholds = thresholds.shape.as_list()[0]
    if multi_label:
        one_thresh = math_ops.equal(math_ops.cast(1, dtype=dtypes.int32), array_ops.rank(thresholds), name='one_set_of_thresholds_cond')
    else:
        [y_pred, y_true], _ = ragged_assert_compatible_and_get_flat_values([y_pred, y_true], sample_weight)
        one_thresh = math_ops.cast(True, dtype=dtypes.bool)
    invalid_keys = [key for key in variables_to_update if key not in list(ConfusionMatrix)]
    if invalid_keys:
        raise ValueError('Invalid keys: {}. Valid variable key options are: "{}"'.format(invalid_keys, list(ConfusionMatrix)))
    with ops.control_dependencies([check_ops.assert_greater_equal(y_pred, math_ops.cast(0.0, dtype=y_pred.dtype), message='predictions must be >= 0'), check_ops.assert_less_equal(y_pred, math_ops.cast(1.0, dtype=y_pred.dtype), message='predictions must be <= 1')]):
        if sample_weight is None:
            y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(y_pred, y_true)
        else:
            sample_weight = math_ops.cast(sample_weight, dtype=variable_dtype)
            y_pred, y_true, sample_weight = losses_utils.squeeze_or_expand_dimensions(y_pred, y_true, sample_weight=sample_weight)
    y_pred.shape.assert_is_compatible_with(y_true.shape)
    if top_k is not None:
        y_pred = _filter_top_k(y_pred, top_k)
    if class_id is not None:
        y_true = y_true[..., class_id]
        y_pred = y_pred[..., class_id]
    if thresholds_distributed_evenly and compat.forward_compatible(2021, 6, 8):
        return _update_confusion_matrix_variables_optimized(variables_to_update, y_true, y_pred, thresholds, multi_label=multi_label, sample_weights=sample_weight, label_weights=label_weights, thresholds_with_epsilon=thresholds_with_epsilon)
    pred_shape = array_ops.shape(y_pred)
    num_predictions = pred_shape[0]
    if y_pred.shape.ndims == 1:
        num_labels = 1
    else:
        num_labels = gen_math_ops.Prod(input=pred_shape[1:], axis=0)
    thresh_label_tile = array_ops.where_v2(one_thresh, num_labels, array_ops.ones([], dtype=dtypes.int32))
    if multi_label:
        predictions_extra_dim = array_ops.expand_dims(y_pred, 0)
        labels_extra_dim = array_ops.expand_dims(math_ops.cast(y_true, dtype=dtypes.bool), 0)
    else:
        predictions_extra_dim = array_ops.reshape(y_pred, [1, -1])
        labels_extra_dim = array_ops.reshape(math_ops.cast(y_true, dtype=dtypes.bool), [1, -1])
    if multi_label:
        thresh_pretile_shape = [num_thresholds, 1, -1]
        thresh_tiles = [1, num_predictions, thresh_label_tile]
        data_tiles = [num_thresholds, 1, 1]
    else:
        thresh_pretile_shape = [num_thresholds, -1]
        thresh_tiles = [1, num_predictions * num_labels]
        data_tiles = [num_thresholds, 1]
    thresh_tiled = array_ops.tile(array_ops.reshape(thresholds, thresh_pretile_shape), array_ops_stack.stack(thresh_tiles))
    preds_tiled = array_ops.tile(predictions_extra_dim, data_tiles)
    pred_is_pos = math_ops.greater(preds_tiled, thresh_tiled)
    label_is_pos = array_ops.tile(labels_extra_dim, data_tiles)
    if sample_weight is not None:
        sample_weight = weights_broadcast_ops.broadcast_weights(math_ops.cast(sample_weight, dtype=variable_dtype), y_pred)
        weights_tiled = array_ops.tile(array_ops.reshape(sample_weight, thresh_tiles), data_tiles)
    else:
        weights_tiled = None
    if label_weights is not None and (not multi_label):
        label_weights = array_ops.expand_dims(label_weights, 0)
        label_weights = weights_broadcast_ops.broadcast_weights(label_weights, y_pred)
        label_weights_tiled = array_ops.tile(array_ops.reshape(label_weights, thresh_tiles), data_tiles)
        if weights_tiled is None:
            weights_tiled = label_weights_tiled
        else:
            weights_tiled = math_ops.multiply(weights_tiled, label_weights_tiled)
    update_ops = []

    def weighted_assign_add(label, pred, weights, var):
        label_and_pred = math_ops.cast(math_ops.logical_and(label, pred), dtype=var.dtype)
        if weights is not None:
            label_and_pred *= math_ops.cast(weights, dtype=var.dtype)
        return var.assign_add(math_ops.reduce_sum(label_and_pred, 1))
    loop_vars = {ConfusionMatrix.TRUE_POSITIVES: (label_is_pos, pred_is_pos)}
    update_tn = ConfusionMatrix.TRUE_NEGATIVES in variables_to_update
    update_fp = ConfusionMatrix.FALSE_POSITIVES in variables_to_update
    update_fn = ConfusionMatrix.FALSE_NEGATIVES in variables_to_update
    if update_fn or update_tn:
        pred_is_neg = math_ops.logical_not(pred_is_pos)
        loop_vars[ConfusionMatrix.FALSE_NEGATIVES] = (label_is_pos, pred_is_neg)
    if update_fp or update_tn:
        label_is_neg = math_ops.logical_not(label_is_pos)
        loop_vars[ConfusionMatrix.FALSE_POSITIVES] = (label_is_neg, pred_is_pos)
        if update_tn:
            loop_vars[ConfusionMatrix.TRUE_NEGATIVES] = (label_is_neg, pred_is_neg)
    for matrix_cond, (label, pred) in loop_vars.items():
        if matrix_cond in variables_to_update:
            update_ops.append(weighted_assign_add(label, pred, weights_tiled, variables_to_update[matrix_cond]))
    return control_flow_ops.group(update_ops)