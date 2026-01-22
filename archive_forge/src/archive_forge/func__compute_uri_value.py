import os
import yaml
from mlflow.exceptions import ExecutionException
from mlflow.projects import env_type
from mlflow.tracking import artifact_utils
from mlflow.utils import data_utils
from mlflow.utils.environment import _PYTHON_ENV_FILE_NAME
from mlflow.utils.file_utils import get_local_path_or_none
from mlflow.utils.string_utils import is_string_type, quote
def _compute_uri_value(self, user_param_value):
    if not data_utils.is_uri(user_param_value):
        raise ExecutionException(f'Expected URI for parameter {self.name} but got {user_param_value}')
    return user_param_value