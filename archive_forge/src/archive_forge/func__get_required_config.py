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
@classmethod
def _get_required_config(cls, dataset_config: Dict[str, Any], key: str) -> Any:
    """
        Obtains the value associated with the specified dataset configuration key, first verifying
        that the key is present in the config and throwing if it is not.

        Args:
            dataset_config: Dictionary representation of the recipe dataset configuration
                (i.e. the `data` section of recipe.yaml).
            key: The key within the dataset configuration for which to fetch the associated
                value.

        Returns:
            The value associated with the specified configuration key.
        """
    try:
        return dataset_config[key]
    except KeyError:
        raise MlflowException(f"The `{key}` configuration key must be specified for dataset with using '{dataset_config.get('using')}' format") from None