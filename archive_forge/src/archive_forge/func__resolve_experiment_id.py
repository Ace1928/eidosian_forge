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
def _resolve_experiment_id(experiment_name=None, experiment_id=None):
    """
    Resolve experiment.

    Verifies either one or other is specified - cannot be both selected.

    If ``experiment_name`` is provided and does not exist, an experiment
    of that name is created and its id is returned.

    Args:
        experiment_name: Name of experiment under which to launch the run.
        experiment_id: ID of experiment under which to launch the run.

    Returns:
        str
    """
    if experiment_name and experiment_id:
        raise MlflowException("Specify only one of 'experiment_name' or 'experiment_id'.")
    if experiment_id:
        return str(experiment_id)
    if experiment_name:
        client = tracking.MlflowClient()
        exp = client.get_experiment_by_name(experiment_name)
        if exp:
            return exp.experiment_id
        else:
            _logger.info("'%s' does not exist. Creating a new experiment", experiment_name)
            return client.create_experiment(experiment_name)
    return _get_experiment_id()