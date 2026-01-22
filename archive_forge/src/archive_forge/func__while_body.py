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
def _while_body(iteration_number, loss_ta, mean_ta, covariance_ta):
    """Perform a processing step on a single window of data."""
    base_offset = iteration_number * self.output_window_size
    model_outputs = self._process_window(features={feature_name: feature_value[:, base_offset:base_offset + self.window_size] for feature_name, feature_value in features.items()}, mode=mode, exogenous_regressors=exogenous_regressors[:, base_offset:base_offset + self.window_size])
    assert len(model_outputs.predictions) == 3
    assert 'mean' in model_outputs.predictions
    assert 'covariance' in model_outputs.predictions
    assert 'observed' in model_outputs.predictions
    return (iteration_number + 1, loss_ta.write(iteration_number, model_outputs.loss), mean_ta.write(iteration_number, model_outputs.predictions['mean']), covariance_ta.write(iteration_number, model_outputs.predictions['covariance']))