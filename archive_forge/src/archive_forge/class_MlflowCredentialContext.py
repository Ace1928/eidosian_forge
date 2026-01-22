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
class MlflowCredentialContext:
    """Sets and clears credentials on a context using the provided profile URL."""

    def __init__(self, databricks_profile_url):
        self.databricks_profile_url = databricks_profile_url or 'databricks'
        self.db_utils = _get_dbutils()

    def __enter__(self):
        db_creds = get_databricks_host_creds(self.databricks_profile_url)
        self.db_utils.notebook.entry_point.putMlflowProperties(db_creds.host, db_creds.ignore_tls_verification, db_creds.token, db_creds.username, db_creds.password)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.db_utils.notebook.entry_point.clearMlflowProperties()