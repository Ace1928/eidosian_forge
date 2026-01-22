import logging
import os
import platform
import posixpath
import subprocess
import sys
from pathlib import Path
import mlflow
from mlflow import tracking
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.projects import env_type
from mlflow.projects.backend.abstract_backend import AbstractBackend
from mlflow.projects.submitted_run import LocalSubmittedRun
from mlflow.projects.utils import (
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.azure_blob_artifact_repo import AzureBlobArtifactRepository
from mlflow.store.artifact.gcs_artifact_repo import GCSArtifactRepository
from mlflow.store.artifact.hdfs_artifact_repo import HdfsArtifactRepository
from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.utils import env_manager as _EnvManager
from mlflow.utils.conda import get_or_create_conda_env
from mlflow.utils.databricks_utils import get_databricks_env_vars, is_in_databricks_runtime
from mlflow.utils.environment import _PythonEnv
from mlflow.utils.file_utils import get_or_create_nfs_tmp_dir
from mlflow.utils.mlflow_tags import MLFLOW_PROJECT_ENV
from mlflow.utils.os import is_windows
from mlflow.utils.virtualenv import (
def _invoke_mlflow_run_subprocess(work_dir, entry_point, parameters, experiment_id, env_manager, docker_args, storage_dir, run_id):
    """
    Run an MLflow project asynchronously by invoking ``mlflow run`` in a subprocess, returning
    a SubmittedRun that can be used to query run status.
    """
    _logger.info('=== Asynchronously launching MLflow run with ID %s ===', run_id)
    mlflow_run_arr = _build_mlflow_run_cmd(uri=work_dir, entry_point=entry_point, docker_args=docker_args, storage_dir=storage_dir, env_manager=env_manager, run_id=run_id, parameters=parameters)
    env_vars = get_run_env_vars(run_id, experiment_id)
    env_vars.update(get_databricks_env_vars(mlflow.get_tracking_uri()))
    mlflow_run_subprocess = _run_mlflow_run_cmd(mlflow_run_arr, env_vars)
    return LocalSubmittedRun(run_id, mlflow_run_subprocess)