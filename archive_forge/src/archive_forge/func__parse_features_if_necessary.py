from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clustering_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_output
def _parse_features_if_necessary(features, feature_columns):
    """Helper function to convert the input points into a usable format.

  Args:
    features: The input features.
    feature_columns: An optionable iterable containing all the feature columns
      used by the model. All items in the set should be feature column instances
      that can be passed to `tf.feature_column.input_layer`. If this is None,
      all features will be used.

  Returns:
    If `features` is a dict of `k` features (optionally filtered by
    `feature_columns`), each of which is a vector of `n` scalars, the return
    value is a Tensor of shape `(n, k)` representing `n` input points, where the
    items in the `k` dimension are sorted lexicographically by `features` key.
    If `features` is not a dict, it is returned unmodified.
  """
    if not isinstance(features, dict):
        return features
    if feature_columns:
        return tf.compat.v1.feature_column.input_layer(features, feature_columns)
    keys = sorted(features.keys())
    with ops.colocate_with(features[keys[0]]):
        return tf.concat([features[k] for k in keys], axis=1)