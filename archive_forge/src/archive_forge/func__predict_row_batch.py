import collections
import functools
import importlib
import inspect
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import warnings
from copy import deepcopy
from functools import lru_cache
from typing import Any, Dict, Iterator, Optional, Tuple, Union
import numpy as np
import pandas
import yaml
import mlflow
import mlflow.pyfunc.loaders
import mlflow.pyfunc.model
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.flavor_backend_registry import get_flavor_backend
from mlflow.models.model import _DATABRICKS_FS_LOADER_MODULE, MLMODEL_FILE_NAME
from mlflow.models.signature import (
from mlflow.models.utils import (
from mlflow.protos.databricks_pb2 import (
from mlflow.pyfunc.model import (
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types.llm import (
from mlflow.utils import (
from mlflow.utils import env_manager as _EnvManager
from mlflow.utils._spark_utils import modified_environ
from mlflow.utils.annotations import deprecated, developer_stable, experimental
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import (
from mlflow.utils.model_utils import (
from mlflow.utils.nfs_on_spark import get_nfs_cache_root_dir
from mlflow.utils.requirements_utils import (
def _predict_row_batch(predict_fn, args):
    input_schema = model_metadata.get_input_schema()
    args = list(args)
    if len(args) == 1 and isinstance(args[0], pandas.DataFrame):
        pdf = args[0]
    else:
        if input_schema is None:
            names = [str(i) for i in range(len(args))]
        else:
            names = input_schema.input_names()
            required_names = input_schema.required_input_names()
            if len(args) > len(names):
                args = args[:len(names)]
            if len(args) < len(required_names):
                raise MlflowException('Model input is missing required columns. Expected {} required input columns {}, but the model received only {} unnamed input columns (Since the columns were passed unnamed they are expected to be in the order specified by the schema).'.format(len(names), names, len(args)))
        pdf = pandas.DataFrame(data={names[i]: arg if isinstance(arg, pandas.Series) else arg.apply(lambda row: row.to_dict(), axis=1) for i, arg in enumerate(args)}, columns=names)
    result = predict_fn(pdf, params)
    if isinstance(result, dict):
        result = {k: list(v) for k, v in result.items()}
    if isinstance(result_type, ArrayType) and isinstance(result_type.elementType, ArrayType):
        result_values = _convert_array_values(result, result_type)
        return pandas.Series(result_values)
    if not isinstance(result, pandas.DataFrame):
        result = pandas.DataFrame([result]) if np.isscalar(result) else pandas.DataFrame(result)
    if isinstance(result_type, SparkStructType):
        return _convert_struct_values(result, result_type)
    elem_type = result_type.elementType if isinstance(result_type, ArrayType) else result_type
    if type(elem_type) == IntegerType:
        result = result.select_dtypes([np.byte, np.ubyte, np.short, np.ushort, np.int32]).astype(np.int32)
    elif type(elem_type) == LongType:
        result = result.select_dtypes([np.byte, np.ubyte, np.short, np.ushort, int]).astype(np.int64)
    elif type(elem_type) == FloatType:
        result = result.select_dtypes(include=(np.number,)).astype(np.float32)
    elif type(elem_type) == DoubleType:
        result = result.select_dtypes(include=(np.number,)).astype(np.float64)
    elif type(elem_type) == BooleanType:
        result = result.select_dtypes([bool, np.bool_]).astype(bool)
    if len(result.columns) == 0:
        raise MlflowException(message=f"The model did not produce any values compatible with the requested type '{elem_type}'. Consider requesting udf with StringType or Arraytype(StringType).", error_code=INVALID_PARAMETER_VALUE)
    if type(elem_type) == StringType:
        result = result.applymap(str)
    if type(result_type) == ArrayType:
        return pandas.Series(result.to_numpy().tolist())
    else:
        return result[result.columns[0]]