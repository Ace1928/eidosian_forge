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
def _predict_ops(self, features):
    """Add ops for prediction to the graph."""
    with tf.compat.v1.variable_scope('model', use_resource=True):
        prediction = self.model.predict(features=features)
    prediction[feature_keys.PredictionResults.TIMES] = features[feature_keys.PredictionFeatures.TIMES]
    return estimator_lib.EstimatorSpec(predictions=prediction, mode=estimator_lib.ModeKeys.PREDICT)