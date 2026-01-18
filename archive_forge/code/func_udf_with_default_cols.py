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
@functools.wraps(udf)
def udf_with_default_cols(*args):
    if len(args) == 0:
        input_schema = model_metadata.get_input_schema()
        if input_schema and len(input_schema.optional_input_names()) > 0:
            raise MlflowException(message='Cannot apply UDF without column names specified when model signature contains optional columns.', error_code=INVALID_PARAMETER_VALUE)
        if input_schema and len(input_schema.inputs) > 0:
            if input_schema.has_input_names():
                input_names = input_schema.input_names()
                return udf(*input_names)
            else:
                raise MlflowException(message='Cannot apply udf because no column names specified. The udf expects {} columns with types: {}. Input column names could not be inferred from the model signature (column names not found).'.format(len(input_schema.inputs), input_schema.inputs), error_code=INVALID_PARAMETER_VALUE)
        else:
            raise MlflowException('Attempting to apply udf on zero columns because no column names were specified as arguments or inferred from the model signature.', error_code=INVALID_PARAMETER_VALUE)
    else:
        return udf(*args)