from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import distributions
from tensorflow.python.ops import gen_math_ops
from tensorflow_estimator.python.estimator import estimator_lib
from tensorflow_estimator.python.estimator.canned.timeseries import model
from tensorflow_estimator.python.estimator.canned.timeseries import model_utils
from tensorflow_estimator.python.estimator.canned.timeseries.feature_keys import PredictionFeatures
from tensorflow_estimator.python.estimator.canned.timeseries.feature_keys import TrainEvalFeatures
def _parse_ta(values_ta):
    """Helper function to parse the returned TensorArrays."""
    if not isinstance(values_ta, tf.TensorArray):
        return None
    predictions_length = prediction_iterations * self.output_window_size
    values_packed = values_ta.stack()
    output_values = tf.reshape(tf.compat.v1.transpose(values_packed, [1, 0, 2, 3]), tf.stack([batch_size, predictions_length, -1]))
    return output_values[:, :num_predict_values, :]