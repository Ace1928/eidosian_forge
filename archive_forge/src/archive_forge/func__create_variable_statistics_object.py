from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import tensorflow as tf
from tensorflow.python.ops import gen_math_ops
from tensorflow_estimator.python.estimator.canned.timeseries.feature_keys import TrainEvalFeatures
def _create_variable_statistics_object(self):
    """Creates non-trainable variables representing input statistics."""
    series_start_moments = Moments(mean=tf.compat.v1.get_variable(name='series_start_mean', shape=[self._num_features], dtype=self._dtype, initializer=tf.compat.v1.initializers.zeros(), trainable=False), variance=tf.compat.v1.get_variable(name='series_start_variance', shape=[self._num_features], dtype=self._dtype, initializer=tf.compat.v1.initializers.ones(), trainable=False))
    overall_feature_moments = Moments(mean=tf.compat.v1.get_variable(name='overall_feature_mean', shape=[self._num_features], dtype=self._dtype, initializer=tf.compat.v1.initializers.zeros(), trainable=False), variance=tf.compat.v1.get_variable(name='overall_feature_var', shape=[self._num_features], dtype=self._dtype, initializer=tf.compat.v1.initializers.ones(), trainable=False))
    start_time = tf.compat.v1.get_variable(name='start_time', dtype=tf.dtypes.int64, initializer=tf.dtypes.int64.max, trainable=False)
    total_observation_count = tf.compat.v1.get_variable(name='total_observation_count', shape=[], dtype=tf.dtypes.int64, initializer=tf.compat.v1.initializers.ones(), trainable=False)
    return InputStatistics(series_start_moments=series_start_moments, overall_feature_moments=overall_feature_moments, start_time=start_time, total_observation_count=total_observation_count)