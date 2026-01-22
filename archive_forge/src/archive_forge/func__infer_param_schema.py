import logging
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.types import DataType
from mlflow.types.schema import (
def _infer_param_schema(parameters: Dict[str, Any]):
    if not isinstance(parameters, dict):
        raise MlflowException.invalid_parameter_value(f'Expected parameters to be dict, got {type(parameters).__name__}')
    param_specs = []
    invalid_params = []
    for name, value in parameters.items():
        try:
            value_type, shape = _infer_type_and_shape(value)
            param_specs.append(ParamSpec(name=name, dtype=value_type, default=value, shape=shape))
        except Exception as e:
            invalid_params.append((name, value, e))
    if invalid_params:
        raise MlflowException.invalid_parameter_value(f'Failed to infer schema for parameters: {invalid_params}')
    return ParamSchema(param_specs)