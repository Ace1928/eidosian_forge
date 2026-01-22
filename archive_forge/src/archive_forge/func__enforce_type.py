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
def _enforce_type(data: Any, data_type: Union[DataType, Array, Object, Map], required=True):
    if isinstance(data_type, DataType):
        return _enforce_datatype(data, data_type, required=required)
    if isinstance(data_type, Array):
        return _enforce_array(data, data_type, required=required)
    if isinstance(data_type, Object):
        return _enforce_object(data, data_type, required=required)
    if isinstance(data_type, Map):
        return _enforce_map(data, data_type, required=required)
    raise MlflowException(f'Invalid data type: {data_type!r}')