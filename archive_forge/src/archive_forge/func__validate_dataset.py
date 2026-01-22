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
def _validate_dataset(dataset: Dataset):
    if dataset is None:
        raise MlflowException('Dataset cannot be None', INVALID_PARAMETER_VALUE)
    if dataset.name is None:
        raise MlflowException('Dataset name cannot be None', INVALID_PARAMETER_VALUE)
    if dataset.digest is None:
        raise MlflowException('Dataset digest cannot be None', INVALID_PARAMETER_VALUE)
    if dataset.source_type is None:
        raise MlflowException('Dataset source_type cannot be None', INVALID_PARAMETER_VALUE)
    if dataset.source is None:
        raise MlflowException('Dataset source cannot be None', INVALID_PARAMETER_VALUE)
    if len(dataset.name) > MAX_DATASET_NAME_SIZE:
        raise MlflowException(f'Dataset name exceeds the maximum length of {MAX_DATASET_NAME_SIZE}', INVALID_PARAMETER_VALUE)
    if len(dataset.digest) > MAX_DATASET_DIGEST_SIZE:
        raise MlflowException(f'Dataset digest exceeds the maximum length of {MAX_DATASET_DIGEST_SIZE}', INVALID_PARAMETER_VALUE)
    if len(dataset.source) > MAX_DATASET_SOURCE_SIZE:
        raise MlflowException(f'Dataset source exceeds the maximum length of {MAX_DATASET_SOURCE_SIZE}', INVALID_PARAMETER_VALUE)
    if dataset.schema is not None and len(dataset.schema) > MAX_DATASET_SCHEMA_SIZE:
        raise MlflowException(f'Dataset schema exceeds the maximum length of {MAX_DATASET_SCHEMA_SIZE}', INVALID_PARAMETER_VALUE)
    if dataset.profile is not None and len(dataset.profile) > MAX_DATASET_PROFILE_SIZE:
        raise MlflowException(f'Dataset profile exceeds the maximum length of {MAX_DATASET_PROFILE_SIZE}', INVALID_PARAMETER_VALUE)