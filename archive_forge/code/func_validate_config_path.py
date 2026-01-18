import json
import sys
from inspect import signature
import click
from mlflow.deployments import interface
from mlflow.environment_variables import MLFLOW_DEPLOYMENTS_CONFIG
from mlflow.utils import cli_args
from mlflow.utils.annotations import experimental
from mlflow.utils.proto_json_utils import NumpyEncoder, _get_jsonable_obj
def validate_config_path(_ctx, _param, value):
    from mlflow.gateway.config import _validate_config
    try:
        _validate_config(value)
        return value
    except Exception as e:
        raise click.BadParameter(str(e))