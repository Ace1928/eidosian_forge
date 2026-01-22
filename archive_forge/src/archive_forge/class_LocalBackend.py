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
class LocalBackend(AbstractBackend):

    def run(self, project_uri, entry_point, params, version, backend_config, tracking_uri, experiment_id):
        work_dir = fetch_and_validate_project(project_uri, version, entry_point, params)
        project = load_project(work_dir)
        if MLFLOW_LOCAL_BACKEND_RUN_ID_CONFIG in backend_config:
            run_id = backend_config[MLFLOW_LOCAL_BACKEND_RUN_ID_CONFIG]
        else:
            run_id = None
        active_run = get_or_create_run(run_id, project_uri, experiment_id, work_dir, version, entry_point, params)
        command_args = []
        command_separator = ' '
        env_manager = backend_config[PROJECT_ENV_MANAGER]
        synchronous = backend_config[PROJECT_SYNCHRONOUS]
        docker_args = backend_config[PROJECT_DOCKER_ARGS]
        storage_dir = backend_config[PROJECT_STORAGE_DIR]
        build_image = backend_config[PROJECT_BUILD_IMAGE]
        docker_auth = backend_config[PROJECT_DOCKER_AUTH]
        if env_manager is None:
            env_manager = _env_type_to_env_manager(project.env_type)
        elif project.env_type == env_type.PYTHON and env_manager == _EnvManager.CONDA:
            raise MlflowException.invalid_parameter_value("python_env project cannot be executed using conda. Set `--env-manager` to 'virtualenv' or 'local' to execute this project.")
        if project.docker_env:
            from mlflow.projects.docker import build_docker_image, validate_docker_env, validate_docker_installation
            tracking.MlflowClient().set_tag(active_run.info.run_id, MLFLOW_PROJECT_ENV, 'docker')
            validate_docker_env(project)
            validate_docker_installation()
            image = build_docker_image(work_dir=work_dir, repository_uri=project.name, base_image=project.docker_env.get('image'), run_id=active_run.info.run_id, build_image=build_image, docker_auth=docker_auth)
            command_args += _get_docker_command(image=image, active_run=active_run, docker_args=docker_args, volumes=project.docker_env.get('volumes'), user_env_vars=project.docker_env.get('environment'))
        elif env_manager == _EnvManager.VIRTUALENV:
            tracking.MlflowClient().set_tag(active_run.info.run_id, MLFLOW_PROJECT_ENV, 'virtualenv')
            command_separator = ' && '
            if project.env_type == env_type.CONDA:
                python_env = _PythonEnv.from_conda_yaml(project.env_config_path)
            else:
                python_env = _PythonEnv.from_yaml(project.env_config_path) if project.env_config_path else _PythonEnv()
            if is_in_databricks_runtime():
                nfs_tmp_dir = get_or_create_nfs_tmp_dir()
                env_root = Path(nfs_tmp_dir) / 'envs'
                pyenv_root = env_root / _PYENV_ROOT_DIR
                virtualenv_root = env_root / _VIRTUALENV_ENVS_DIR
                env_vars = _get_virtualenv_extra_env_vars(str(env_root))
            else:
                pyenv_root = None
                virtualenv_root = Path(_get_mlflow_virtualenv_root())
                env_vars = None
            python_bin_path = _install_python(python_env.python, pyenv_root=pyenv_root)
            work_dir_path = Path(work_dir)
            env_name = _get_virtualenv_name(python_env, work_dir_path)
            env_dir = virtualenv_root / env_name
            activate_cmd = _create_virtualenv(work_dir_path, python_bin_path, env_dir, python_env, extra_env=env_vars)
            command_args += [activate_cmd]
        elif env_manager == _EnvManager.CONDA:
            tracking.MlflowClient().set_tag(active_run.info.run_id, MLFLOW_PROJECT_ENV, 'conda')
            command_separator = ' && '
            conda_env = get_or_create_conda_env(project.env_config_path)
            command_args += conda_env.get_activate_command()
        if synchronous:
            command_args += get_entry_point_command(project, entry_point, params, storage_dir)
            command_str = command_separator.join(command_args)
            return _run_entry_point(command_str, work_dir, experiment_id, run_id=active_run.info.run_id)
        return _invoke_mlflow_run_subprocess(work_dir=work_dir, entry_point=entry_point, parameters=params, experiment_id=experiment_id, env_manager=env_manager, docker_args=docker_args, storage_dir=storage_dir, run_id=active_run.info.run_id)