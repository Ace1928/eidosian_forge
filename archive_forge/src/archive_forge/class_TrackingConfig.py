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
class TrackingConfig:
    """
    The MLflow Tracking configuration associated with an MLflow Recipe, including the
    Tracking URI and information about the destination Experiment for writing results.
    """
    _KEY_TRACKING_URI = 'mlflow_tracking_uri'
    _KEY_EXPERIMENT_NAME = 'mlflow_experiment_name'
    _KEY_EXPERIMENT_ID = 'mlflow_experiment_id'
    _KEY_RUN_NAME = 'mlflow_run_name'
    _KEY_ARTIFACT_LOCATION = 'mlflow_experiment_artifact_location'

    def __init__(self, tracking_uri: str, experiment_name: Optional[str]=None, experiment_id: Optional[str]=None, run_name: Optional[str]=None, artifact_location: Optional[str]=None):
        """
        Args:
            tracking_uri: The MLflow Tracking URI.
            experiment_name: The MLflow Experiment name. At least one of ``experiment_name`` or
                ``experiment_id`` must be specified. If both are specified, they must be consistent
                with Tracking server state. Note that this Experiment may not exist prior to recipe
                execution.
            experiment_id: The MLflow Experiment ID. At least one of ``experiment_name`` or
                ``experiment_id`` must be specified. If both are specified, they must be consistent
                with Tracking server state. Note that this Experiment may not exist prior to recipe
                execution.
            run_name: The MLflow Run Name. If the run name is not specified, then a random name is
                set for the run.
            artifact_location: The artifact location to use for the Experiment, if the Experiment
                does not already exist. If the Experiment already exists, this location is ignored.
        """
        if tracking_uri is None:
            raise MlflowException(message='`tracking_uri` must be specified', error_code=INVALID_PARAMETER_VALUE)
        if (experiment_name, experiment_id).count(None) != 1:
            raise MlflowException(message='Exactly one of `experiment_name` or `experiment_id` must be specified', error_code=INVALID_PARAMETER_VALUE)
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.experiment_id = experiment_id
        self.run_name = run_name
        self.artifact_location = artifact_location

    def to_dict(self) -> Dict[str, str]:
        """
        Obtains a dictionary representation of the MLflow Tracking configuration.

        Returns:
            A dictionary representation of the MLflow Tracking configuration.
        """
        config_dict = {TrackingConfig._KEY_TRACKING_URI: self.tracking_uri}
        if self.experiment_name:
            config_dict[TrackingConfig._KEY_EXPERIMENT_NAME] = self.experiment_name
        elif self.experiment_id:
            config_dict[TrackingConfig._KEY_EXPERIMENT_ID] = self.experiment_id
        if self.artifact_location:
            config_dict[TrackingConfig._KEY_ARTIFACT_LOCATION] = self.artifact_location
        if self.run_name:
            config_dict[TrackingConfig._KEY_RUN_NAME] = self.run_name
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, str]) -> 'TrackingConfig':
        """
        Creates a ``TrackingConfig`` instance from a dictionary representation.

        Args:
            config_dict: A dictionary representation of the MLflow Tracking configuration.

        Returns:
            A ``TrackingConfig`` instance.
        """
        return TrackingConfig(tracking_uri=config_dict.get(TrackingConfig._KEY_TRACKING_URI), experiment_name=config_dict.get(TrackingConfig._KEY_EXPERIMENT_NAME), experiment_id=config_dict.get(TrackingConfig._KEY_EXPERIMENT_ID), run_name=config_dict.get(TrackingConfig._KEY_RUN_NAME), artifact_location=config_dict.get(TrackingConfig._KEY_ARTIFACT_LOCATION))