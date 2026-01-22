import abc
from typing import Dict, Optional, TypeVar, Union
import numpy as np
import pandas as pd
from ray.air.util.data_batch_conversion import (
from ray.train.predictor import Predictor
from ray.util.annotations import DeveloperAPI
def _predict_numpy(self, data: Union[np.ndarray, Dict[str, np.ndarray]], dtype: Optional[Union[TensorDtype, Dict[str, TensorDtype]]]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    if isinstance(data, dict) and len(data) == 1:
        data = next(iter(data.values()))
    model_input = self._arrays_to_tensors(data, dtype)
    model_output = self.call_model(model_input)
    if isinstance(model_output, dict):
        return {k: self._tensor_to_array(v) for k, v in model_output.items()}
    else:
        return {'predictions': self._tensor_to_array(model_output)}