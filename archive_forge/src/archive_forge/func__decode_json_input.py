import inspect
import json
import logging
import os
import shlex
import sys
import traceback
from typing import Dict, NamedTuple, Optional, Tuple
import flask
from mlflow.environment_variables import MLFLOW_SCORING_SERVER_REQUEST_TIMEOUT
from mlflow.exceptions import MlflowException
from mlflow.pyfunc.model import _log_warning_if_params_not_in_predict_signature
from mlflow.types import ParamSchema, Schema
from mlflow.utils import reraise
from mlflow.utils.annotations import deprecated
from mlflow.utils.file_utils import path_to_local_file_uri
from mlflow.utils.os import is_windows
from mlflow.utils.proto_json_utils import (
from mlflow.version import VERSION
from io import StringIO
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INVALID_PARAMETER_VALUE
from mlflow.server.handlers import catch_mlflow_exception
def _decode_json_input(json_input):
    """
    Args:
        json_input: A JSON-formatted string representation of TF serving input or a Pandas
                    DataFrame, or a stream containing such a string representation.

    Returns:
        A dictionary representation of the JSON input.
    """
    if isinstance(json_input, dict):
        return json_input
    try:
        decoded_input = json.loads(json_input)
    except json.decoder.JSONDecodeError as ex:
        raise MlflowInvalidInputException(f"Ensure that input is a valid JSON formatted string. Error: '{ex!r}'\nInput: \n{json_input}\n") from ex
    if isinstance(decoded_input, dict):
        return decoded_input
    if isinstance(decoded_input, list):
        raise MlflowInvalidInputException(f'{REQUIRED_INPUT_FORMAT}. Received a list.')
    raise MlflowInvalidInputException(f"{REQUIRED_INPUT_FORMAT}. Received unexpected input type '{type(decoded_input)}.")