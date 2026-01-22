import logging
import os
import pathlib
import posixpath
from typing import Any, Dict, Optional
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.file_utils import read_yaml, render_and_merge_yaml
def _verify_is_recipe_root_directory(recipe_root_path: str) -> str:
    """
    Verifies that the specified local filesystem path is the path of a recipe root directory.

    Args:
        recipe_root_path: The absolute path of the recipe root directory on the local
            filesystem to validate.

    Raises:
        MlflowException: If the specified ``recipe_root_path`` is not a recipe root
            directory.
    """
    recipe_yaml_path = os.path.join(recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
    if not os.path.exists(recipe_yaml_path):
        raise MlflowException(f'Failed to find {_RECIPE_CONFIG_FILE_NAME} in {recipe_yaml_path}!')