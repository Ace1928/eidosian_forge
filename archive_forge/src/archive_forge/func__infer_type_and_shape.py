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
def _infer_type_and_shape(value):
    if isinstance(value, (list, np.ndarray)):
        ndim = _get_array_depth(value)
        if ndim != 1:
            raise MlflowException.invalid_parameter_value(f'Expected parameters to be 1D array or scalar, got {ndim}D array')
        if all((DataType.is_datetime(v) for v in value)):
            return (DataType.datetime, (-1,))
        value_type = _infer_numpy_dtype(np.array(value).dtype)
        return (value_type, (-1,))
    elif DataType.is_datetime(value):
        return (DataType.datetime, None)
    elif np.isscalar(value):
        try:
            value_type = _infer_numpy_dtype(np.array(value).dtype)
            return (value_type, None)
        except (Exception, MlflowException) as e:
            raise MlflowException.invalid_parameter_value(f'Failed to infer schema for parameter {value}: {e!r}')
    raise MlflowException.invalid_parameter_value(f'Expected parameters to be 1D array or scalar, got {type(value).__name__}')