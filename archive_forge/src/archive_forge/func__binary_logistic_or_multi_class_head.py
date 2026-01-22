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
def _binary_logistic_or_multi_class_head(n_classes, weight_column, label_vocabulary, loss_reduction):
    """Creates either binary or multi-class head.

  Args:
    n_classes: Number of label classes.
    weight_column: A string or a `_NumericColumn` created by
      `tf.feature_column.numeric_column` defining feature column representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example. If it is a string, it is
      used as a key to fetch weight tensor from the `features`. If it is a
      `_NumericColumn`, raw tensor is fetched by key `weight_column.key`, then
      weight_column.normalizer_fn is applied on it to get weight tensor.
    label_vocabulary: A list of strings represents possible label values. If
      given, labels must be string type and have any value in
      `label_vocabulary`. If it is not given, that means labels are already
      encoded as integer or float within [0, 1] for `n_classes=2` and encoded as
      integer values in {0, 1,..., n_classes-1} for `n_classes`>2 . Also there
      will be errors if vocabulary is not provided and labels are string.
    loss_reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch. Defaults to `SUM`.

  Returns:
    `head._Head` instance.
  """
    if n_classes == 2:
        head = _binary_logistic_head_with_sigmoid_cross_entropy_loss(weight_column=weight_column, label_vocabulary=label_vocabulary, loss_reduction=loss_reduction)
    else:
        head = _multi_class_head_with_softmax_cross_entropy_loss(n_classes, weight_column=weight_column, label_vocabulary=label_vocabulary, loss_reduction=loss_reduction)
    return head