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
def _validate_max_retries(max_retries):
    max_retry_limit = _MLFLOW_HTTP_REQUEST_MAX_RETRIES_LIMIT.get()
    if max_retry_limit < 0:
        raise MlflowException(message=f'The current maximum retry limit is invalid ({max_retry_limit}). Cannot be negative.', error_code=INVALID_PARAMETER_VALUE)
    if max_retries >= max_retry_limit:
        raise MlflowException(message=f'The configured max_retries value ({max_retries}) is in excess of the maximum allowable retries ({max_retry_limit})', error_code=INVALID_PARAMETER_VALUE)
    if max_retries < 0:
        raise MlflowException(message=f'The max_retries value must be either 0 a positive integer. Got {max_retries}', error_code=INVALID_PARAMETER_VALUE)