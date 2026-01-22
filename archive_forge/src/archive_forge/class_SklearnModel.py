import logging
import os
from .. import prediction_utils
from .._interfaces import PredictionClient
import numpy as np
from ..prediction_utils import DEFAULT_MODEL_FILE_NAME_JOBLIB
from ..prediction_utils import DEFAULT_MODEL_FILE_NAME_PICKLE
from ..prediction_utils import load_joblib_or_pickle_model
from ..prediction_utils import PredictionError
class SklearnModel(prediction_utils.BaseModel):
    """The implementation of Scikit-learn Model.
  """

    def predict(self, instances, stats=None, **kwargs):
        """Override the predict method to remove TF-specific args from kwargs."""
        kwargs.pop(prediction_utils.SIGNATURE_KEY, None)
        return super(SklearnModel, self).predict(instances, stats, **kwargs)

    def preprocess(self, instances, stats=None, **kwargs):
        return instances

    def postprocess(self, predicted_outputs, original_input=None, stats=None, **kwargs):
        if isinstance(predicted_outputs, np.ndarray):
            return predicted_outputs.tolist()
        if isinstance(predicted_outputs, list):
            return predicted_outputs
        raise PredictionError(PredictionError.INVALID_OUTPUTS, 'Bad output type returned.The predict function should return either a numpy ndarray or a list.')