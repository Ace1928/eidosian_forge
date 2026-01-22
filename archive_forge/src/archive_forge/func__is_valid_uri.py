import urllib
from typing import Optional
from urllib.parse import urlparse
from mlflow.environment_variables import MLFLOW_DEPLOYMENTS_TARGET
from mlflow.exceptions import MlflowException
from mlflow.utils.uri import append_to_uri_path
def _is_valid_uri(uri: str) -> bool:
    """
    Evaluates the basic structure of a provided uri to determine if the scheme and
    netloc are provided
    """
    try:
        parsed = urlparse(uri)
        return bool(parsed.scheme and parsed.netloc)
    except ValueError:
        return False