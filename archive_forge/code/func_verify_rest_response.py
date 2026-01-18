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
def verify_rest_response(response, endpoint):
    """Verify the return code and format, raise exception if the request was not successful."""
    if response.status_code != 200:
        if _can_parse_as_json_object(response.text):
            raise RestException(json.loads(response.text))
        else:
            base_msg = f'API request to endpoint {endpoint} failed with error code {response.status_code} != 200'
            raise MlflowException(f"{base_msg}. Response body: '{response.text}'", error_code=get_error_code(response.status_code))
    if endpoint.startswith(_REST_API_PATH_PREFIX) and (not _can_parse_as_json_object(response.text)):
        base_msg = 'API request to endpoint was successful but the response body was not in a valid JSON format'
        raise MlflowException(f"{base_msg}. Response body: '{response.text}'")
    return response