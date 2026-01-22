import inspect
import json
import os
import pydoc  # used for importing python classes from their FQN
import sys
from ._interfaces import Model
from .prediction_utils import PredictionError
def _validate_prediction_class(user_class):
    """Validates a user provided implementation of Model class.

  Args:
    user_class: The user provided custom Model class.

  Raises:
    PredictionError: for any of the following:
      (1) the user model class does not have the correct method signatures for
      the predict method
  """
    user_class_name = user_class.__name__
    if not hasattr(user_class, 'from_path'):
        raise PredictionError(PredictionError.INVALID_USER_CODE, 'User provided model class %s must implement the from_path method.' % user_class_name)
    if not hasattr(user_class, 'predict'):
        raise PredictionError(PredictionError.INVALID_USER_CODE, 'The provided model class, %s, is missing the required predict method.' % user_class_name)
    if sys.version_info.major == 2:
        user_signature = inspect.getargspec(user_class.predict).args
        model_signature = inspect.getargspec(Model.predict).args
    else:
        user_signature = inspect.getfullargspec(user_class.predict).args
        model_signature = inspect.getfullargspec(Model.predict).args
    user_predict_num_args = len(user_signature)
    predict_num_args = len(model_signature)
    if predict_num_args is not user_predict_num_args:
        raise PredictionError(PredictionError.INVALID_USER_CODE, 'The provided model class, %s, has a predict method with an invalid signature. Expected signature: %s User signature: %s' % (user_class_name, model_signature, user_signature))