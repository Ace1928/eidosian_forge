import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import yaml
import mlflow
from mlflow import pyfunc
from mlflow.models import Model, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
class _PromptflowModelWrapper:

    def __init__(self, model, model_config: Optional[Dict[str, Any]]=None):
        from promptflow._sdk._mlflow import FlowInvoker
        self.model = model
        model_config = model_config or {}
        connection_provider = model_config.get(_CONNECTION_PROVIDER_CONFIG_KEY, 'local')
        _logger.info('Using connection provider: %s', connection_provider)
        connection_overrides = model_config.get(_CONNECTION_OVERRIDES_CONFIG_KEY, None)
        _logger.info('Using connection overrides: %s', connection_overrides)
        self.model_invoker = FlowInvoker(self.model, connection_provider=connection_provider, connections_name_overrides=connection_overrides)

    def predict(self, data: Union[pd.DataFrame, List[Union[str, Dict[str, Any]]]], params: Optional[Dict[str, Any]]=None) -> Union[dict, list]:
        """
        Args:
            data: Model input data. Either a pandas DataFrame with only 1 row or a dictionary.

                     .. code-block:: python
                        loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)
                        # Predict on a flow input dictionary.
                        print(loaded_model.predict({"text": "Python Hello World!"}))

            params: Additional parameters to pass to the model for inference.

                       .. Note:: Experimental: This parameter may change or be removed in a future
                                               release without warning.

        Returns
            Model predictions. Dict type, example ``{"output": "

print('Hello World!')"}``
        """
        if isinstance(data, pd.DataFrame):
            messages = data.to_dict(orient='records')
            if len(messages) > 1:
                raise mlflow.MlflowException.invalid_parameter_value(_INVALID_PREDICT_INPUT_ERROR_MESSAGE)
            messages = messages[0]
            return [self.model_invoker.invoke(messages)]
        elif isinstance(data, dict):
            messages = data
            return self.model_invoker.invoke(messages)
        raise mlflow.MlflowException.invalid_parameter_value(_INVALID_PREDICT_INPUT_ERROR_MESSAGE)