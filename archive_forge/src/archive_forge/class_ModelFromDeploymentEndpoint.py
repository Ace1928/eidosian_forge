import json
import keyword
import logging
import math
import operator
import os
import pathlib
import signal
import struct
import sys
import urllib
import urllib.parse
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from decimal import Decimal
from types import FunctionType
from typing import Any, Dict, Optional
import mlflow
from mlflow.data.dataset import Dataset
from mlflow.entities import RunTag
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag
from mlflow.exceptions import MlflowException
from mlflow.models.evaluation.validation import (
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking.client import MlflowClient
from mlflow.utils import _get_fully_qualified_class_name, insecure_hash
from mlflow.utils.annotations import developer_stable, experimental
from mlflow.utils.class_utils import _get_class_from_string
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import MLFLOW_DATASET_CONTEXT
from mlflow.utils.proto_json_utils import NumpyEncoder
from mlflow.utils.string_utils import generate_feature_name_if_not_string
class ModelFromDeploymentEndpoint(mlflow.pyfunc.PythonModel):

    def __init__(self, endpoint, params):
        self.endpoint = endpoint
        self.params = params

    def predict(self, context, model_input: pd.DataFrame):
        if len(model_input.columns) != 1:
            raise MlflowException(f'The number of input columns must be 1, but got {model_input.columns}. Multi-column input is not supported for evaluating an MLflow Deployments endpoint. Please include the input text or payload in a single column.', error_code=INVALID_PARAMETER_VALUE)
        input_column = model_input.columns[0]
        predictions = []
        for data in model_input[input_column]:
            if isinstance(data, str):
                prediction = _call_deployments_api(self.endpoint, data, self.params)
            elif isinstance(data, dict):
                prediction = _call_deployments_api(self.endpoint, data, self.params, wrap_payload=False)
            else:
                raise MlflowException(f'Invalid input column type: {type(data)}. The input data must be either a string or a dictionary contains the request payload for evaluating an MLflow Deployments endpoint.', error_code=INVALID_PARAMETER_VALUE)
            predictions.append(prediction)
        return pd.Series(predictions)