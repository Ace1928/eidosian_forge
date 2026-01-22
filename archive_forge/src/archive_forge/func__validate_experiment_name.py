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
def _validate_experiment_name(experiment_name):
    """Check that `experiment_name` is a valid string and raise an exception if it isn't."""
    if experiment_name == '' or experiment_name is None:
        raise MlflowException(f"Invalid experiment name: '{experiment_name}'", error_code=INVALID_PARAMETER_VALUE)
    if not is_string_type(experiment_name):
        raise MlflowException(f'Invalid experiment name: {experiment_name}. Expects a string.', error_code=INVALID_PARAMETER_VALUE)