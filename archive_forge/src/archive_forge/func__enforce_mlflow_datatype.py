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
def _enforce_mlflow_datatype(name, values: pd.Series, t: DataType):
    """
    Enforce the input column type matches the declared in model input schema.

    The following type conversions are allowed:

    1. object -> string
    2. int -> long (upcast)
    3. float -> double (upcast)
    4. int -> double (safe conversion)
    5. np.datetime64[x] -> datetime (any precision)
    6. object -> datetime

    NB: pandas does not have native decimal data type, when user train and infer
    model from pyspark dataframe that contains decimal type, the schema will be
    treated as float64.
    7. decimal -> double

    Any other type mismatch will raise error.
    """
    if values.dtype == object and t not in (DataType.binary, DataType.string):
        values = values.infer_objects()
    if t == DataType.string and values.dtype == object:
        return values
    if t.to_pandas() == values.dtype or t.to_numpy() == values.dtype:
        return values
    if t == DataType.binary and values.dtype.kind == t.binary.to_numpy().kind:
        return values
    if t == DataType.datetime and values.dtype.kind == t.to_numpy().kind:
        return values.astype(np.dtype('datetime64[ns]'))
    if t == DataType.datetime and (values.dtype == object or values.dtype == t.to_python()):
        try:
            return values.astype(np.dtype('datetime64[ns]'), errors='raise')
        except ValueError as e:
            raise MlflowException(f'Failed to convert column {name} from type {values.dtype} to {t}.') from e
    if t == DataType.boolean and values.dtype == object:
        return values
    if t == DataType.double and values.dtype == decimal.Decimal:
        try:
            return pd.to_numeric(values, errors='raise')
        except ValueError:
            raise MlflowException(f'Failed to convert column {name} from type {values.dtype} to {t}.')
    numpy_type = t.to_numpy()
    if values.dtype.kind == numpy_type.kind:
        is_upcast = values.dtype.itemsize <= numpy_type.itemsize
    elif values.dtype.kind == 'u' and numpy_type.kind == 'i':
        is_upcast = values.dtype.itemsize < numpy_type.itemsize
    elif values.dtype.kind in ('i', 'u') and numpy_type == np.float64:
        is_upcast = values.dtype.itemsize <= 6
    else:
        is_upcast = False
    if is_upcast:
        return values.astype(numpy_type, errors='raise')
    else:

        def all_ints(xs):
            return all((pd.isnull(x) or int(x) == x for x in xs))
        hint = ''
        if values.dtype == np.float64 and numpy_type.kind in ('i', 'u') and values.hasnans and all_ints(values):
            hint = ' Hint: the type mismatch is likely caused by missing values. Integer columns in python can not represent missing values and are therefore encoded as floats. The best way to avoid this problem is to infer the model schema based on a realistic data sample (training dataset) that includes missing values. Alternatively, you can declare integer columns as doubles (float64) whenever these columns may have missing values. See `Handling Integers With Missing Values <https://www.mlflow.org/docs/latest/models.html#handling-integers-with-missing-values>`_ for more details.'
        raise MlflowException(f'Incompatible input types for column {name}. Can not safely convert {values.dtype} to {numpy_type}.{hint}')