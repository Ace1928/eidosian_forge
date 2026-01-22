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
def _predicted_covariance_op(self, activations, num_values):
    activation, activation_size = activations[-1]
    if self.loss == ARModel.NORMAL_LIKELIHOOD_LOSS:
        log_sigma_square = model_utils.fully_connected(activation, activation_size, self.output_window_size * num_values, name='log_sigma_square', activation=None)
        predicted_covariance = gen_math_ops.exp(log_sigma_square)
        predicted_covariance = tf.reshape(predicted_covariance, [-1, self.output_window_size, num_values])
    else:
        shape = tf.stack([tf.compat.v1.shape(activation)[0], tf.constant(self.output_window_size), tf.constant(num_values)])
        predicted_covariance = tf.ones(shape=shape, dtype=activation.dtype)
    return predicted_covariance