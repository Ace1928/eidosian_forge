import configparser
import getpass
import logging
import os
from typing import NamedTuple, Optional, Tuple
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.utils.rest_utils import MlflowHostCreds
def _validate_databricks_auth():
    try:
        from databricks.sdk import WorkspaceClient
    except ImportError:
        raise ImportError('Databricks SDK is not installed. To use `mlflow.login()`, please install databricks-sdk by `pip install databricks-sdk`.')
    try:
        w = WorkspaceClient()
        if 'community' in w.config.host:
            w.clusters.list_zones()
        else:
            w.current_user.me()
        _logger.info(f'Successfully connected to MLflow hosted tracking server! Host: {w.config.host}.')
    except Exception as e:
        raise MlflowException(f'Failed to validate databricks credentials: {e}')