import warnings
import click
from mlflow.environment_variables import MLFLOW_DISABLE_ENV_MANAGER_CONDA_WARNING
from mlflow.utils import env_manager as _EnvManager
def _resolve_env_manager(_, __, env_manager):
    if env_manager is not None:
        _EnvManager.validate(env_manager)
        if env_manager == _EnvManager.CONDA and (not MLFLOW_DISABLE_ENV_MANAGER_CONDA_WARNING.get()):
            warnings.warn(f"Use of conda is discouraged. If you use it, please ensure that your use of conda complies with Anaconda's terms of service (https://legal.anaconda.com/policies/en/?name=terms-of-service). virtualenv is the recommended tool for environment reproducibility. To suppress this warning, set the {MLFLOW_DISABLE_ENV_MANAGER_CONDA_WARNING} environment variable to 'TRUE'.", UserWarning, stacklevel=2)
        return env_manager
    return None