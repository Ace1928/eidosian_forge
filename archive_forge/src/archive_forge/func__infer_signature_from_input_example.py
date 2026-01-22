import inspect
import logging
import re
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, get_type_hints
import numpy as np
import pandas as pd
from mlflow import environment_variables
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import ModelInputExample, _contains_params, _Example
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_DOES_NOT_EXIST
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri, _upload_artifact_to_uri
from mlflow.types.schema import ParamSchema, Schema
from mlflow.types.utils import _infer_param_schema, _infer_schema, _infer_schema_from_type_hint
from mlflow.utils.uri import append_to_uri_path
def _infer_signature_from_input_example(input_example: ModelInputExample, wrapped_model) -> Optional[ModelSignature]:
    """
    Infer the signature from an example input and a PyFunc wrapped model. Catches all exceptions.

    Args:
        input_example: An instance representing a typical input to the model.
        wrapped_model: A PyFunc wrapped model which has a `predict` method.

    Returns:
        A `ModelSignature` object containing the inferred schema of both the model's inputs
        based on the `input_example` and the model's outputs based on the prediction from the
        `wrapped_model`.
    """
    try:
        if _contains_params(input_example):
            input_example, params = input_example
        else:
            params = None
        example = _Example(input_example)
        input_example = deepcopy(example.inference_data)
        input_schema = _infer_schema(input_example)
        params_schema = _infer_param_schema(params) if params else None
        prediction = wrapped_model.predict(input_example, params=params)
        if not input_schema.is_tensor_spec() and isinstance(prediction, np.ndarray) and (prediction.ndim == 1):
            prediction = pd.Series(prediction)
        output_schema = _infer_schema(prediction)
        return ModelSignature(input_schema, output_schema, params_schema)
    except Exception as e:
        if environment_variables._MLFLOW_TESTING.get():
            raise
        _logger.warning(_LOG_MODEL_INFER_SIGNATURE_WARNING_TEMPLATE, repr(e))
        _logger.debug('', exc_info=True)
        return None