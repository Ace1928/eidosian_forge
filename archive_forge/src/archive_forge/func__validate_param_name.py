import logging
import numbers
import posixpath
import re
from typing import List
from mlflow.entities import Dataset, DatasetInput, InputTag, Param, RunTag
from mlflow.environment_variables import MLFLOW_TRUNCATE_LONG_VALUES
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.utils.string_utils import is_string_type
def _validate_param_name(name):
    """Check that `name` is a valid parameter name and raise an exception if it isn't."""
    if name is None:
        raise MlflowException(f'Parameter name cannot be None. {_MISSING_KEY_NAME_MESSAGE}', error_code=INVALID_PARAMETER_VALUE)
    if not _VALID_PARAM_AND_METRIC_NAMES.match(name):
        raise MlflowException(f"Invalid parameter name: '{name}'. {_BAD_CHARACTERS_MESSAGE}", INVALID_PARAMETER_VALUE)
    if path_not_unique(name):
        raise MlflowException(f"Invalid parameter name: '{name}'. {bad_path_message(name)}", INVALID_PARAMETER_VALUE)