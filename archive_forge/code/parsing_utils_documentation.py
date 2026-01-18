from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_lib as fc
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
Adds label and weight spec to given parsing spec.

  Args:
    parsing_spec: A dict mapping each feature key to a `FixedLenFeature` or
      `VarLenFeature` to which label and weight spec are added.
    label_key: A string identifying the label. It means tf.Example stores labels
      with this key.
    label_spec: A `FixedLenFeature`.
    weight_column: A string or a `NumericColumn` created by
      `tf.feature_column.numeric_column` defining feature column representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example. If it is a string, it is
      used as a key to fetch weight tensor from the `features`. If it is a
      `NumericColumn`, raw tensor is fetched by key `weight_column.key`, then
      weight_column.normalizer_fn is applied on it to get weight tensor.

  Returns:
    A dict mapping each feature key to a `FixedLenFeature` or `VarLenFeature`
      value.
  