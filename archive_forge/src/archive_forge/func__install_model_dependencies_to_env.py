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
def _install_model_dependencies_to_env(model_path, env_manager) -> List[str]:
    """:
    Installs model dependencies to the specified environment, which can be either a local
    environment, a conda environment, or a virtualenv.

    Returns:
        Empty list if local environment, otherwise a list of bash commands to activate the
        virtualenv or conda environment.
    """
    model_config_path = os.path.join(model_path, MLMODEL_FILE_NAME)
    model = Model.load(model_config_path)
    conf = model.flavors.get(pyfunc.FLAVOR_NAME, {})
    if pyfunc.ENV not in conf:
        return []
    env_conf = conf[mlflow.pyfunc.ENV]
    if env_manager == em.LOCAL:
        python_env_config_path = os.path.join(model_path, env_conf[em.VIRTUALENV])
        python_env = _PythonEnv.from_yaml(python_env_config_path)
        deps = ' '.join(python_env.build_dependencies + python_env.dependencies)
        deps = deps.replace('requirements.txt', os.path.join(model_path, 'requirements.txt'))
        if Popen(['bash', '-c', f'python -m pip install {deps}']).wait() != 0:
            raise Exception('Failed to install model dependencies.')
        return []
    _logger.info('creating and activating custom environment')
    env = _extract_conda_env(env_conf)
    env_path_dst = os.path.join('/opt/mlflow/', env)
    env_path_dst_dir = os.path.dirname(env_path_dst)
    if not os.path.exists(env_path_dst_dir):
        os.makedirs(env_path_dst_dir)
    shutil.copy2(os.path.join(MODEL_PATH, env), env_path_dst)
    if env_manager == em.CONDA:
        conda_create_model_env = f'conda env create -n custom_env -f {env_path_dst}'
        if Popen(['bash', '-c', conda_create_model_env]).wait() != 0:
            raise Exception('Failed to create model environment.')
        activate_cmd = ['source /miniconda/bin/activate custom_env']
    elif env_manager == em.VIRTUALENV:
        env_activate_cmd = _get_or_create_virtualenv(model_path)
        path = env_activate_cmd.split(' ')[-1]
        os.symlink(path, '/opt/activate')
        activate_cmd = [env_activate_cmd]
    return activate_cmd