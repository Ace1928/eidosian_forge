import logging
import os
from .. import prediction_utils
from .._interfaces import PredictionClient
import numpy as np
from ..prediction_utils import DEFAULT_MODEL_FILE_NAME_JOBLIB
from ..prediction_utils import DEFAULT_MODEL_FILE_NAME_PICKLE
from ..prediction_utils import load_joblib_or_pickle_model
from ..prediction_utils import PredictionError
Override the predict method to remove TF-specific args from kwargs.