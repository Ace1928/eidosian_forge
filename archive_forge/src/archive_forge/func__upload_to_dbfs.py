import hashlib
import json
import logging
import os
import posixpath
import re
import tempfile
import textwrap
import time
from shlex import quote
from mlflow import tracking
from mlflow.entities import RunStatus
from mlflow.environment_variables import MLFLOW_EXPERIMENT_ID, MLFLOW_TRACKING_URI
from mlflow.exceptions import ExecutionException, MlflowException
from mlflow.projects.submitted_run import SubmittedRun
from mlflow.projects.utils import MLFLOW_LOCAL_BACKEND_RUN_ID_CONFIG
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils import databricks_utils, file_utils, rest_utils
from mlflow.utils.mlflow_tags import (
from mlflow.utils.uri import is_databricks_uri, is_http_uri
from mlflow.version import VERSION, is_release_version
def _upload_to_dbfs(self, src_path, dbfs_fuse_uri):
    """
        Upload the file at `src_path` to the specified DBFS URI within the Databricks workspace
        corresponding to the default Databricks CLI profile.
        """
    _logger.info('=== Uploading project to DBFS path %s ===', dbfs_fuse_uri)
    http_endpoint = dbfs_fuse_uri
    with open(src_path, 'rb') as f:
        try:
            self._databricks_api_request(endpoint=http_endpoint, method='POST', data=f)
        except MlflowException as e:
            if 'Error 409' in e.message and 'File already exists' in e.message:
                _logger.info('=== Did not overwrite existing DBFS path %s ===', dbfs_fuse_uri)
            else:
                raise e