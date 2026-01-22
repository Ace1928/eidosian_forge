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
def _check_train_eval_features(features, model):
    """Raise errors if features are not suitable for training/evaluation."""
    if feature_keys.TrainEvalFeatures.TIMES not in features:
        raise ValueError("Expected a '{}' feature for training/evaluation.".format(feature_keys.TrainEvalFeatures.TIMES))
    if feature_keys.TrainEvalFeatures.VALUES not in features:
        raise ValueError("Expected a '{}' feature for training/evaluation.".format(feature_keys.TrainEvalFeatures.VALUES))
    times_feature = features[feature_keys.TrainEvalFeatures.TIMES]
    if not times_feature.get_shape().is_compatible_with([None, None]):
        raise ValueError("Expected shape (batch dimension, window size) for feature '{}' (got shape {})".format(feature_keys.TrainEvalFeatures.TIMES, times_feature.get_shape()))
    values_feature = features[feature_keys.TrainEvalFeatures.VALUES]
    if not values_feature.get_shape().is_compatible_with([None, None, model.num_features]):
        raise ValueError("Expected shape (batch dimension, window size, {num_features}) for feature '{feature_name}', since the model was configured with num_features={num_features} (got shape {got_shape})".format(num_features=model.num_features, feature_name=feature_keys.TrainEvalFeatures.VALUES, got_shape=times_feature.get_shape()))
    _check_feature_shapes_compatible_with(features=features, compatible_with_name=feature_keys.TrainEvalFeatures.TIMES, compatible_with_value=times_feature, ignore=set([feature_keys.State.STATE_TUPLE]))