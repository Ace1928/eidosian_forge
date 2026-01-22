from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import estimator_lib
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned.timeseries import feature_keys
from tensorflow_estimator.python.estimator.export import export_lib
def _serving_ops(self, features):
    """Add ops for serving to the graph."""
    with tf.compat.v1.variable_scope('model', use_resource=True):
        filtering_features = {}
        prediction_features = {}
        values_length = tf.compat.v1.shape(features[feature_keys.FilteringFeatures.VALUES])[1]
        for key, value in features.items():
            if key == feature_keys.State.STATE_TUPLE:
                continue
            if key == feature_keys.FilteringFeatures.VALUES:
                filtering_features[key] = value
            else:
                filtering_features[key] = value[:, :values_length]
                prediction_features[key] = value[:, values_length:]
        cold_filtering_outputs = self.model.define_loss(features=filtering_features, mode=estimator_lib.ModeKeys.EVAL)
        prediction_features[feature_keys.State.STATE_TUPLE] = cold_filtering_outputs.end_state
    with tf.compat.v1.variable_scope('model', reuse=True):
        prediction_outputs = self.model.predict(features=prediction_features)
    return estimator_lib.EstimatorSpec(mode=estimator_lib.ModeKeys.PREDICT, export_outputs={feature_keys.SavedModelLabels.PREDICT: _NoStatePredictOutput(prediction_outputs)}, predictions={})