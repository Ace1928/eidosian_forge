import base64
import json
import requests
from mlflow.environment_variables import (
from mlflow.exceptions import (
from mlflow.protos import databricks_pb2
from mlflow.protos.databricks_pb2 import ENDPOINT_NOT_FOUND, INVALID_PARAMETER_VALUE, ErrorCode
from mlflow.utils.proto_json_utils import parse_dict
from mlflow.utils.request_utils import (
from mlflow.utils.string_utils import strip_suffix
def _validate_backoff_factor(backoff_factor):
    max_backoff_factor_limit = _MLFLOW_HTTP_REQUEST_MAX_BACKOFF_FACTOR_LIMIT.get()
    if max_backoff_factor_limit < 0:
        raise MlflowException(message=f'The current maximum backoff factor limit is invalid ({max_backoff_factor_limit}). Cannot be negative.', error_code=INVALID_PARAMETER_VALUE)
    if backoff_factor >= max_backoff_factor_limit:
        raise MlflowException(message=f'The configured backoff_factor value ({backoff_factor}) is in excess of the maximum allowable backoff_factor limit ({max_backoff_factor_limit})', error_code=INVALID_PARAMETER_VALUE)
    if backoff_factor < 0:
        raise MlflowException(message=f'The backoff_factor value must be either 0 a positive integer. Got {backoff_factor}', error_code=INVALID_PARAMETER_VALUE)