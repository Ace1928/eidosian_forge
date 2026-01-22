import json
import logging
import os
import yaml
import mlflow.projects.databricks
import mlflow.utils.uri
from mlflow import tracking
from mlflow.entities import RunStatus
from mlflow.exceptions import ExecutionException, MlflowException
from mlflow.projects.backend import loader
from mlflow.projects.submitted_run import SubmittedRun
from mlflow.projects.utils import (
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils import env_manager as _EnvManager
from mlflow.utils.mlflow_tags import (
def _maybe_set_run_terminated(active_run, status):
    """
    If the passed-in active run is defined and still running (i.e. hasn't already been terminated
    within user code), mark it as terminated with the passed-in status.
    """
    if active_run is None:
        return
    run_id = active_run.info.run_id
    cur_status = tracking.MlflowClient().get_run(run_id).info.status
    if RunStatus.is_terminated(cur_status):
        return
    tracking.MlflowClient().set_terminated(run_id, status)