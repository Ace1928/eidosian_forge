from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import collections
import six
import tensorflow as tf
from tensorflow_estimator.python.estimator.canned.timeseries import math_utils
from tensorflow_estimator.python.estimator.canned.timeseries.feature_keys import TrainEvalFeatures
def _get_exogenous_embedding_shape(self):
    """Computes the shape of the vector returned by _process_exogenous_features.

    Returns:
      The shape as a list. Does not include a batch dimension.
    """
    if not self._exogenous_feature_columns:
        return (0,)
    with tf.Graph().as_default():
        parsed_features = tf.compat.v1.feature_column.make_parse_example_spec(self._exogenous_feature_columns)
        placeholder_features = tf.compat.v1.io.parse_example(serialized=tf.compat.v1.placeholder(shape=[None], dtype=tf.dtypes.string), features=parsed_features)
        embedded = tf.compat.v1.feature_column.input_layer(features=placeholder_features, feature_columns=self._exogenous_feature_columns)
        return embedded.get_shape().as_list()[1:]