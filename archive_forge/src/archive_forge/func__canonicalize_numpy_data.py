from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy
from tensorflow.python.util.all_util import remove_undocumented
from tensorflow_estimator.python.estimator.canned.timeseries import feature_keys as _feature_keys
from tensorflow_estimator.python.estimator.canned.timeseries import head as _head
from tensorflow_estimator.python.estimator.canned.timeseries import model_utils as _model_utils
def _canonicalize_numpy_data(data, require_single_batch):
    """Do basic checking and reshaping for Numpy data.

  Args:
    data: A dictionary mapping keys to Numpy arrays, with several possible
      shapes (requires keys `TrainEvalFeatures.TIMES` and
      `TrainEvalFeatures.VALUES`): Single example; `TIMES` is a scalar and
        `VALUES` is either a scalar or a vector of length [number of features].
        Sequence; `TIMES` is a vector of shape [series length], `VALUES` either
        has shape [series length] (univariate) or [series length x number of
        features] (multivariate). Batch of sequences; `TIMES` is a vector of
        shape [batch size x series length], `VALUES` has shape [batch size x
        series length] or [batch size x series length x number of features]. In
        any case, `VALUES` and any exogenous features must have their shapes
        prefixed by the shape of the value corresponding to the `TIMES` key.
    require_single_batch: If True, raises an error if the provided data has a
      batch dimension > 1.

  Returns:
    A dictionary with features normalized to have shapes prefixed with [batch
    size x series length]. The sizes of dimensions which were omitted in the
    inputs are 1.
  Raises:
    ValueError: If dimensions are incorrect or do not match, or required
      features are missing.
  """
    features = {key: numpy.array(value) for key, value in data.items()}
    if _feature_keys.TrainEvalFeatures.TIMES not in features or _feature_keys.TrainEvalFeatures.VALUES not in features:
        raise ValueError('{} and {} are required features.'.format(_feature_keys.TrainEvalFeatures.TIMES, _feature_keys.TrainEvalFeatures.VALUES))
    times = features[_feature_keys.TrainEvalFeatures.TIMES]
    for key, value in features.items():
        if value.shape[:len(times.shape)] != times.shape:
            raise ValueError("All features must have their shapes prefixed by the shape of the times feature. Got shape {} for feature '{}', but shape {} for '{}'".format(value.shape, key, times.shape, _feature_keys.TrainEvalFeatures.TIMES))
    if not times.shape:
        if not features[_feature_keys.TrainEvalFeatures.VALUES].shape:
            features[_feature_keys.TrainEvalFeatures.VALUES] = features[_feature_keys.TrainEvalFeatures.VALUES][..., None]
        elif len(features[_feature_keys.TrainEvalFeatures.VALUES].shape) > 1:
            raise ValueError("Got an unexpected number of dimensions for the '{}' feature. Was expecting at most 1 dimension ([number of features]) since '{}' does not have a batch or time dimension, but got shape {}".format(_feature_keys.TrainEvalFeatures.VALUES, _feature_keys.TrainEvalFeatures.TIMES, features[_feature_keys.TrainEvalFeatures.VALUES].shape))
        features = {key: value[None, None, ...] for key, value in features.items()}
    if len(times.shape) == 1:
        if len(features[_feature_keys.TrainEvalFeatures.VALUES].shape) == 1:
            features[_feature_keys.TrainEvalFeatures.VALUES] = features[_feature_keys.TrainEvalFeatures.VALUES][..., None]
        elif len(features[_feature_keys.TrainEvalFeatures.VALUES].shape) > 2:
            raise ValueError("Got an unexpected number of dimensions for the '{}' feature. Was expecting at most 2 dimensions ([series length, number of features]) since '{}' does not have a batch dimension, but got shape {}".format(_feature_keys.TrainEvalFeatures.VALUES, _feature_keys.TrainEvalFeatures.TIMES, features[_feature_keys.TrainEvalFeatures.VALUES].shape))
        features = {key: value[None, ...] for key, value in features.items()}
    elif len(features[_feature_keys.TrainEvalFeatures.TIMES].shape) != 2:
        raise ValueError('Got an unexpected number of dimensions for times. Was expecting at most two ([batch size, series length]), but got shape {}.'.format(times.shape))
    if require_single_batch:
        if features[_feature_keys.TrainEvalFeatures.TIMES].shape[0] != 1:
            raise ValueError('Got batch input, was expecting unbatched input.')
    return features