import logging
import os
from .. import prediction_utils
from .._interfaces import PredictionClient
import numpy as np
from ..prediction_utils import DEFAULT_MODEL_FILE_NAME_JOBLIB
from ..prediction_utils import DEFAULT_MODEL_FILE_NAME_PICKLE
from ..prediction_utils import load_joblib_or_pickle_model
from ..prediction_utils import PredictionError
class SklearnClient(PredictionClient):
    """A loaded scikit-learn model to be used for prediction."""

    def __init__(self, predictor):
        self._predictor = predictor

    def predict(self, inputs, stats=None, **kwargs):
        stats = stats or prediction_utils.Stats()
        stats[prediction_utils.FRAMEWORK] = prediction_utils.SCIKIT_LEARN_FRAMEWORK_NAME
        stats[prediction_utils.ENGINE] = prediction_utils.SCIKIT_LEARN_FRAMEWORK_NAME
        with stats.time(prediction_utils.SESSION_RUN_TIME):
            try:
                return self._predictor.predict(inputs, **kwargs)
            except Exception as e:
                logging.exception('Exception while predicting with sklearn model.')
                raise PredictionError(PredictionError.FAILED_TO_RUN_MODEL, 'Exception during sklearn prediction: ' + str(e))