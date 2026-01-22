import logging
import os
from typing import Any, Dict, Optional
import yaml
import mlflow
from mlflow import pyfunc
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.autologging_utils import autologging_integration, safe_patch
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
class _PaddleWrapper:
    """
    Wrapper class that creates a predict function such that
    predict(data: pd.DataFrame) -> model's output as pd.DataFrame (pandas DataFrame)
    """

    def __init__(self, pd_model):
        self.pd_model = pd_model

    def predict(self, data, params: Optional[Dict[str, Any]]=None):
        """
        Args:
            data: Model input data.
            params: Additional parameters to pass to the model for inference.

                .. Note:: Experimental: This parameter may change or be removed in a future
                           release without warning.

        Returns:
            Model predictions.
        """
        import numpy as np
        import paddle
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            inp_data = data.values.astype(np.float32)
        elif isinstance(data, np.ndarray):
            inp_data = data
        elif isinstance(data, (list, dict)):
            raise TypeError('The paddle flavor does not support List or Dict input types. Please use a pandas.DataFrame or a numpy.ndarray')
        else:
            raise TypeError('Input data should be pandas.DataFrame or numpy.ndarray')
        inp_data = np.squeeze(inp_data)
        self.pd_model.eval()
        predicted = self.pd_model(paddle.to_tensor(inp_data))
        return pd.DataFrame(predicted.numpy())