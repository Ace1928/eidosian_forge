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
def _validate_experiment_artifact_location(artifact_location):
    if artifact_location is not None and artifact_location.startswith('runs:'):
        raise MlflowException(f"Artifact location cannot be a runs:/ URI. Given: '{artifact_location}'", error_code=INVALID_PARAMETER_VALUE)