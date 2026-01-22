import json
import logging
import pathlib
import shutil
import tempfile
import uuid
from typing import Any, Dict, Optional
import mlflow
from mlflow.environment_variables import MLFLOW_RUN_CONTEXT
from mlflow.exceptions import MlflowException, RestException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.recipes.utils import get_recipe_name
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.context.registry import resolve_tags
from mlflow.tracking.default_experiment import DEFAULT_EXPERIMENT_ID
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.tracking.fluent import set_experiment as fluent_set_experiment
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.file_utils import path_to_local_file_uri, path_to_local_sqlite_uri
from mlflow.utils.git_utils import get_git_branch, get_git_commit, get_git_repo_url
from mlflow.utils.mlflow_tags import (
def apply_recipe_tracking_config(tracking_config: TrackingConfig):
    """
    Applies the specified ``TrackingConfig`` in the current context by setting the associated
    MLflow Tracking URI (via ``mlflow.set_tracking_uri()``) and setting the associated MLflow
    Experiment (via ``mlflow.set_experiment()``), creating it if necessary.

    Args:
        tracking_config: The MLflow Recipe ``TrackingConfig`` to apply.
    """
    mlflow.set_tracking_uri(uri=tracking_config.tracking_uri)
    client = MlflowClient()
    if tracking_config.experiment_name is not None:
        experiment = client.get_experiment_by_name(name=tracking_config.experiment_name)
        if not experiment:
            _logger.info("Experiment with name '%s' does not exist. Creating a new experiment.", tracking_config.experiment_name)
            try:
                client.create_experiment(name=tracking_config.experiment_name, artifact_location=tracking_config.artifact_location)
            except RestException:
                raise MlflowException(f'Could not create an MLflow Experiment with name {tracking_config.experiment_name}. Please create an MLflow Experiment for this recipe and specify its name in the "name" field of the "experiment" section in your profile configuration.')
    fluent_set_experiment(experiment_id=tracking_config.experiment_id, experiment_name=tracking_config.experiment_name)