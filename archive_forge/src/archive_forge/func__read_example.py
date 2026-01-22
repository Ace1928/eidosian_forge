import datetime as dt
import decimal
import json
import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
from mlflow.models import Model
from mlflow.store.artifact.utils.models import get_model_name_and_version
from mlflow.types import DataType, ParamSchema, ParamSpec, Schema, TensorSpec
from mlflow.types.schema import Array, Map, Object, Property
from mlflow.types.utils import (
from mlflow.utils.annotations import experimental
from mlflow.utils.proto_json_utils import (
from mlflow.utils.uri import get_databricks_profile_uri_from_artifact_uri
def _read_example(mlflow_model: Model, path: str):
    """
    Read example from a model directory. Returns None if there is no example metadata (i.e. the
    model was saved without example). Raises FileNotFoundError if there is model metadata but the
    example file is missing.

    Args:
        mlflow_model: Model metadata.
        path: Path to the model directory.

    Returns:
        Input example data or None if the model has no example.
    """
    input_example = _get_mlflow_model_input_example_dict(mlflow_model, path)
    if input_example is None:
        return None
    example_type = mlflow_model.saved_input_example_info['type']
    input_schema = mlflow_model.signature.inputs if mlflow_model.signature is not None else None
    if mlflow_model.saved_input_example_info.get(EXAMPLE_PARAMS_KEY, None):
        input_example = input_example[EXAMPLE_DATA_KEY]
    if example_type == 'json_object':
        return input_example
    if example_type == 'ndarray':
        return _read_tensor_input_from_json(input_example, schema=input_schema)
    if example_type in ['sparse_matrix_csc', 'sparse_matrix_csr']:
        return _read_sparse_matrix_from_json(input_example, example_type)
    return dataframe_from_parsed_json(input_example, pandas_orient='split', schema=input_schema)