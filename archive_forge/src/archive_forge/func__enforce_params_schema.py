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
def _enforce_params_schema(params: Optional[Dict[str, Any]], schema: Optional[ParamSchema]):
    if schema is None:
        if params in [None, {}]:
            return params
        params_info = f'Ignoring provided params: {list(params.keys())}' if isinstance(params, dict) else 'Ignoring invalid params (not a dictionary).'
        _logger.warning(f'`params` can only be specified at inference time if the model signature defines a params schema. This model does not define a params schema. {params_info}')
        return {}
    params = {} if params is None else params
    if not isinstance(params, dict):
        raise MlflowException.invalid_parameter_value(f"Parameters must be a dictionary. Got type '{type(params).__name__}'.")
    if not isinstance(schema, ParamSchema):
        raise MlflowException.invalid_parameter_value(f"Parameters schema must be an instance of ParamSchema. Got type '{type(schema).__name__}'.")
    if any((not isinstance(k, str) for k in params.keys())):
        _logger.warning('Keys in parameters should be of type `str`, but received non-string keys.Converting all keys to string...')
        params = {str(k): v for k, v in params.items()}
    allowed_keys = {param.name for param in schema.params}
    ignored_keys = set(params) - allowed_keys
    if ignored_keys:
        _logger.warning(f'Unrecognized params {list(ignored_keys)} are ignored for inference. Supported params are: {allowed_keys}. To enable them, please add corresponding schema in ModelSignature.')
    params = {k: params[k] for k in params if k in allowed_keys}
    invalid_params = set()
    for param_spec in schema.params:
        if param_spec.name in params:
            try:
                params[param_spec.name] = ParamSpec.validate_param_spec(params[param_spec.name], param_spec)
            except MlflowException as e:
                invalid_params.add((param_spec.name, e.message))
        else:
            params[param_spec.name] = param_spec.default
    if invalid_params:
        raise MlflowException.invalid_parameter_value(f'Invalid parameters found: {invalid_params!r}')
    return params