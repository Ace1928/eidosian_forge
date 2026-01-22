import os
import platform
import shutil
import subprocess
import sys
import yaml
import mlflow
from mlflow import MlflowClient
from mlflow.environment_variables import MLFLOW_WHEELED_MODEL_PIP_DOWNLOAD_OPTIONS
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST
from mlflow.pyfunc.model import MLMODEL_FILE_NAME, Model
from mlflow.store.artifact.utils.models import _parse_model_uri, get_model_name_and_version
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.annotations import experimental
from mlflow.utils.environment import (
from mlflow.utils.model_utils import _validate_and_prepare_target_save_path
from mlflow.utils.uri import get_databricks_profile_uri_from_artifact_uri
def _update_conda_env(self, new_pip_deps, conda_env_path):
    """
        Updates the list pip packages in the conda.yaml file to the list of wheels in the wheels
        directory.
        {
            "name": "env",
            "channels": [...],
            "dependencies": [
                ...,
                "pip",
                {"pip": [...]},  <- Overwrite this with list of wheels
            ],
        }

        Args:
            new_pip_deps: List of pip dependencies as wheels
            conda_env_path: Path to conda.yaml file in the model directory
        """
    with open(conda_env_path) as f:
        conda_env = yaml.safe_load(f)
    new_conda_env = _overwrite_pip_deps(conda_env, new_pip_deps)
    with open(conda_env_path, 'w') as out:
        yaml.safe_dump(new_conda_env, stream=out, default_flow_style=False)