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
def _process_window(self, features, mode, exogenous_regressors):
    """Compute model outputs on a single window of data."""
    times = tf.cast(features[TrainEvalFeatures.TIMES], tf.dtypes.int64)
    values = tf.cast(features[TrainEvalFeatures.VALUES], dtype=self.dtype)
    exogenous_regressors = tf.cast(exogenous_regressors, dtype=self.dtype)
    original_values = values
    expected_times_shape = [None, self.window_size]
    if not times.get_shape().is_compatible_with(expected_times_shape):
        raise ValueError("ARModel with input_window_size={input_window_size} and output_window_size={output_window_size} expects feature '{times_feature}' to have shape (batch_size, {window_size}) (for any batch_size), but got shape {times_shape}. If you are using RandomWindowInputFn, set window_size={window_size} or adjust the input_window_size and output_window_size arguments to ARModel.".format(input_window_size=self.input_window_size, output_window_size=self.output_window_size, times_feature=TrainEvalFeatures.TIMES, window_size=self.window_size, times_shape=times.get_shape()))
    values = self._scale_data(values)
    if self.input_window_size > 0:
        input_values = values[:, :self.input_window_size, :]
    else:
        input_values = None
    prediction_ops = self.prediction_ops(times, input_values, exogenous_regressors)
    prediction = prediction_ops['mean']
    covariance = prediction_ops['covariance']
    targets = tf.slice(values, [0, self.input_window_size, 0], [-1, -1, -1])
    targets.get_shape().assert_is_compatible_with(prediction.get_shape())
    if mode == estimator_lib.ModeKeys.EVAL and self.loss == ARModel.SQUARED_LOSS:
        loss = self.loss_op(self._scale_back_data(targets), {'mean': self._scale_back_data(prediction_ops['mean'])})
    else:
        loss = self.loss_op(targets, prediction_ops)
    prediction = self._scale_back_data(prediction)
    covariance = self._scale_back_variance(covariance)
    return model.ModelOutputs(loss=loss, end_state=(times[:, -self.input_window_size:], values[:, -self.input_window_size:, :], exogenous_regressors[:, -self.input_window_size:, :]), predictions={'mean': prediction, 'covariance': covariance, 'observed': original_values[:, -self.output_window_size:]}, prediction_times=times[:, -self.output_window_size:])