import os
import yaml
from mlflow.exceptions import ExecutionException
from mlflow.projects import env_type
from mlflow.tracking import artifact_utils
from mlflow.utils import data_utils
from mlflow.utils.environment import _PYTHON_ENV_FILE_NAME
from mlflow.utils.file_utils import get_local_path_or_none
from mlflow.utils.string_utils import is_string_type, quote
@staticmethod
def _sanitize_param_dict(param_dict):
    return {str(key): quote(str(value)) for key, value in param_dict.items()}