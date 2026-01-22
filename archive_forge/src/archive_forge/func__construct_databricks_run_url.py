import functools
import json
import logging
import os
import subprocess
import time
from sys import stderr
from typing import NamedTuple, Optional, TypeVar
import mlflow.utils
from mlflow.environment_variables import MLFLOW_TRACKING_URI
from mlflow.exceptions import MlflowException
from mlflow.legacy_databricks_cli.configure.provider import (
from mlflow.utils._spark_utils import _get_active_spark_session
from mlflow.utils.rest_utils import MlflowHostCreds
from mlflow.utils.uri import get_db_info_from_uri, is_databricks_uri
def _construct_databricks_run_url(host: str, experiment_id: str, run_id: str, workspace_id: Optional[str]=None, artifact_path: Optional[str]=None) -> str:
    run_url = host
    if workspace_id and workspace_id != '0':
        run_url += '?o=' + str(workspace_id)
    run_url += f'#mlflow/experiments/{experiment_id}/runs/{run_id}'
    if artifact_path is not None:
        run_url += f'/artifactPath/{artifact_path.lstrip('/')}'
    return run_url