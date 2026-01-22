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
def _print_description_and_log_tags(self):
    _logger.info('=== Launched MLflow run as Databricks job run with ID %s. Getting run status page URL... ===', self._databricks_run_id)
    run_info = self._job_runner.jobs_runs_get(self._databricks_run_id)
    jobs_page_url = run_info['run_page_url']
    _logger.info("=== Check the run's status at %s ===", jobs_page_url)
    host_creds = databricks_utils.get_databricks_host_creds(self._job_runner.databricks_profile_uri)
    tracking.MlflowClient().set_tag(self._mlflow_run_id, MLFLOW_DATABRICKS_RUN_URL, jobs_page_url)
    tracking.MlflowClient().set_tag(self._mlflow_run_id, MLFLOW_DATABRICKS_SHELL_JOB_RUN_ID, self._databricks_run_id)
    tracking.MlflowClient().set_tag(self._mlflow_run_id, MLFLOW_DATABRICKS_WEBAPP_URL, host_creds.host)
    job_id = run_info.get('job_id')
    if job_id is not None:
        tracking.MlflowClient().set_tag(self._mlflow_run_id, MLFLOW_DATABRICKS_SHELL_JOB_ID, job_id)