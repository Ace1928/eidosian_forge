import logging
import multiprocessing
import os
import shutil
import signal
import sys
from pathlib import Path
from subprocess import Popen, check_call
from typing import List
import mlflow
import mlflow.version
from mlflow import mleap, pyfunc
from mlflow.environment_variables import MLFLOW_DEPLOYMENT_FLAVOR_NAME, MLFLOW_DISABLE_ENV_CREATION
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.pyfunc import _extract_conda_env, mlserver, scoring_server
from mlflow.store.artifact.models_artifact_repo import REGISTERED_MODEL_META_FILE_NAME
from mlflow.utils import env_manager as em
from mlflow.utils.environment import _PythonEnv
from mlflow.utils.file_utils import read_yaml
from mlflow.utils.virtualenv import _get_or_create_virtualenv
from mlflow.version import VERSION as MLFLOW_VERSION
def _install_pyfunc_deps(model_path=None, install_mlflow=False, enable_mlserver=False, env_manager=em.VIRTUALENV):
    """
    Creates a conda env for serving the model at the specified path and installs almost all serving
    dependencies into the environment - MLflow is not installed as it's not available via conda.
    """
    activate_cmd = _install_model_dependencies_to_env(model_path, env_manager) if model_path else []
    server_deps = ['gunicorn[gevent]']
    if enable_mlserver:
        server_deps = ["'mlserver>=1.2.0,!=1.3.1,<1.4.0'", "'mlserver-mlflow>=1.2.0,!=1.3.1,<1.4.0'"]
    install_server_deps = [f'pip install {' '.join(server_deps)}']
    if Popen(['bash', '-c', ' && '.join(activate_cmd + install_server_deps)]).wait() != 0:
        raise Exception('Failed to install serving dependencies into the model environment.')
    if len(activate_cmd) and install_mlflow:
        install_mlflow_cmd = ['pip install /opt/mlflow/.' if _container_includes_mlflow_source() else f'pip install mlflow=={MLFLOW_VERSION}']
        if Popen(['bash', '-c', ' && '.join(activate_cmd + install_mlflow_cmd)]).wait() != 0:
            raise Exception('Failed to install mlflow into the model environment.')
    return activate_cmd