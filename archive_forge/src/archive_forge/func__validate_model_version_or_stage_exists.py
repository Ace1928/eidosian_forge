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
def _validate_model_version_or_stage_exists(version, stage):
    if version and stage:
        raise MlflowException('version and stage cannot be set together', INVALID_PARAMETER_VALUE)
    if not (version or stage):
        raise MlflowException('version or stage must be set', INVALID_PARAMETER_VALUE)