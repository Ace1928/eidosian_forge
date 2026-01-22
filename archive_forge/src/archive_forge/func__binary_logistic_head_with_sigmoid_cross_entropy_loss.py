from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import collections
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column
from tensorflow.python.framework import ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned import prediction_keys
from tensorflow_estimator.python.estimator.export import export_output
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _binary_logistic_head_with_sigmoid_cross_entropy_loss(weight_column=None, thresholds=None, label_vocabulary=None, loss_reduction=tf.compat.v1.losses.Reduction.SUM, loss_fn=None, name=None):
    """Creates a `_Head` for single label binary classification.

  This head uses `sigmoid_cross_entropy_with_logits` loss.

  The head expects `logits` with shape `[D0, D1, ... DN, 1]`.
  In many applications, the shape is `[batch_size, 1]`.

  `labels` must be a dense `Tensor` with shape matching `logits`, namely
  `[D0, D1, ... DN, 1]`. If `label_vocabulary` given, `labels` must be a string
  `Tensor` with values from the vocabulary. If `label_vocabulary` is not given,
  `labels` must be float `Tensor` with values in the interval `[0, 1]`.

  If `weight_column` is specified, weights must be of shape
  `[D0, D1, ... DN]`, or `[D0, D1, ... DN, 1]`.

  The loss is the weighted sum over the input dimensions. Namely, if the input
  labels have shape `[batch_size, 1]`, the loss is the weighted sum over
  `batch_size`.

  Also supports custom `loss_fn`. `loss_fn` takes `(labels, logits)` or
  `(labels, logits, features)` as arguments and returns unreduced loss with
  shape `[D0, D1, ... DN, 1]`. `loss_fn` must support float `labels` with
  shape `[D0, D1, ... DN, 1]`. Namely, the head applies `label_vocabulary` to
  the input labels before passing them to `loss_fn`.

  Args:
    weight_column: A string or a `_NumericColumn` created by
      `tf.feature_column.numeric_column` defining feature column representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example.
    thresholds: Iterable of floats in the range `(0, 1)`. For binary
      classification metrics such as precision and recall, an eval metric is
      generated for each threshold value. This threshold is applied to the
      logistic values to determine the binary classification (i.e., above the
      threshold is `true`, below is `false`.
    label_vocabulary: A list or tuple of strings representing possible label
      values. If it is not given, that means labels are already encoded within
      [0, 1]. If given, labels must be string type and have any value in
      `label_vocabulary`. Note that errors will be raised if `label_vocabulary`
      is not provided but labels are strings.
    loss_reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch. Defaults to `SUM`.
    loss_fn: Optional loss function.
    name: name of the head. If provided, summary and metrics keys will be
      suffixed by `"/" + name`. Also used as `name_scope` when creating ops.

  Returns:
    An instance of `_Head` for binary classification.

  Raises:
    ValueError: If `thresholds` contains a value outside of `(0, 1)`.
    ValueError: If `loss_reduction` is invalid.
    TypeError: if `label_vocabulary` has invalid type.
  """
    thresholds = tuple(thresholds) if thresholds else tuple()
    if label_vocabulary is not None and (not isinstance(label_vocabulary, (list, tuple))):
        raise TypeError('label_vocabulary should be a list or tuple. Given type: {}'.format(type(label_vocabulary)))
    for threshold in thresholds:
        if threshold <= 0.0 or threshold >= 1.0:
            raise ValueError('thresholds not in (0, 1): {}.'.format((thresholds,)))
    if loss_reduction not in tf.compat.v1.losses.Reduction.all() or loss_reduction == tf.compat.v1.losses.Reduction.NONE:
        raise ValueError('Invalid loss_reduction: {}'.format(loss_reduction))
    if loss_fn:
        _validate_loss_fn_args(loss_fn)
    return _BinaryLogisticHeadWithSigmoidCrossEntropyLoss(weight_column=weight_column, thresholds=thresholds, label_vocabulary=label_vocabulary, loss_reduction=loss_reduction, loss_fn=loss_fn, name=name)