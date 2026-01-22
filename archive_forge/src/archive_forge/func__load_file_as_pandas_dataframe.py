import importlib
import logging
import os
import pathlib
import posixpath
import sys
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INVALID_PARAMETER_VALUE
from mlflow.store.artifact.artifact_repo import (
from mlflow.utils._spark_utils import (
from mlflow.utils.file_utils import (
def _load_file_as_pandas_dataframe(self, local_data_file_path: str):
    try:
        sys.path.append(self.recipe_root)
        loader_method = getattr(importlib.import_module(_USER_DEFINED_INGEST_STEP_MODULE), self.loader_method)
    except Exception as e:
        raise MlflowException(message=(f"Failed to import custom dataset loader function '{_USER_DEFINED_INGEST_STEP_MODULE}.{self.loader_method}' for ingesting dataset with format '{self.dataset_format}'.",), error_code=BAD_REQUEST) from e
    try:
        return self._validate_user_code_output(loader_method, local_data_file_path, self.dataset_format)
    except MlflowException as e:
        raise e
    except NotImplementedError:
        raise MlflowException(message=f"Unable to load data file at path '{local_data_file_path}' with format '{self.dataset_format}' using custom loader method '{loader_method.__name__}' because it is not supported. Please update the custom loader method to support this format.", error_code=INVALID_PARAMETER_VALUE) from None
    except Exception as e:
        raise MlflowException(message=f"Unable to load data file at path '{local_data_file_path}' with format '{self.dataset_format}' using custom loader method '{loader_method.__name__}'.", error_code=BAD_REQUEST) from e