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
def _convert_array_values(values, result_type):
    """
    Convert list or numpy array values to spark dataframe column values.
    """
    from pyspark.sql.types import ArrayType, StructType
    if not isinstance(result_type, ArrayType):
        raise MlflowException.invalid_parameter_value(f'result_type must be ArrayType, got {result_type.simpleString()}')
    spark_primitive_type_to_np_type = _get_spark_primitive_type_to_np_type()
    if type(result_type.elementType) in spark_primitive_type_to_np_type:
        np_type = spark_primitive_type_to_np_type[type(result_type.elementType)]
        return None if _is_none_or_nan(values) else np.array(values, dtype=np_type)
    if isinstance(result_type.elementType, ArrayType):
        return [_convert_array_values(v, result_type.elementType) for v in values]
    if isinstance(result_type.elementType, StructType):
        return [_convert_struct_values(v, result_type.elementType) for v in values]
    raise MlflowException.invalid_parameter_value(f'Unsupported array type field with element type {result_type.elementType.simpleString()} in Array type.')