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
def _get_gcs_artifact_cmd_and_envs(artifact_repo):
    cmds = []
    envs = {}
    if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
        credentials_path = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
        cmds = ['-v', f'{credentials_path}:/.gcs']
        envs['GOOGLE_APPLICATION_CREDENTIALS'] = '/.gcs'
    return (cmds, envs)