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
def _enforce_datatype(data: Any, dtype: DataType, required=True):
    if not required and data is None:
        return None
    if not isinstance(dtype, DataType):
        raise MlflowException(f'Expected dtype to be DataType, got {type(dtype).__name__}')
    if not np.isscalar(data):
        raise MlflowException(f'Expected data to be scalar, got {type(data).__name__}')
    pd_series = pd.Series(data)
    try:
        pd_series = _enforce_mlflow_datatype('', pd_series, dtype)
    except MlflowException:
        raise MlflowException(f'Failed to enforce schema of data `{data}` with dtype `{dtype.name}`')
    return pd_series[0]