import abc
import functools
import inspect
import itertools
import typing
import uuid
from abc import abstractmethod
from contextlib import contextmanager
import mlflow
import mlflow.utils.autologging_utils
from mlflow.entities.run_status import RunStatus
from mlflow.environment_variables import _MLFLOW_AUTOLOGGING_TESTING
from mlflow.tracking.client import MlflowClient
from mlflow.utils import gorilla, is_iterator
from mlflow.utils.autologging_utils import _logger
from mlflow.utils.autologging_utils.events import AutologgingEventLogger
from mlflow.utils.autologging_utils.logging_and_warnings import (
from mlflow.utils.mlflow_tags import MLFLOW_AUTOLOGGING
def _validate_autologging_run(autologging_integration, run_id):
    """
    For testing purposes, verifies that an MLflow run produced by an `autologging_integration`
    satisfies the following properties:

        - The run has an autologging tag whose value is the name of the autologging integration
        - The run has a terminal status (e.g., KILLED, FAILED, FINISHED)
    """
    client = MlflowClient()
    run = client.get_run(run_id)
    autologging_tag_value = run.data.tags.get(MLFLOW_AUTOLOGGING)
    assert autologging_tag_value == autologging_integration, f"Autologging run with id {run_id} failed to set autologging tag with expected value. Expected: '{autologging_integration}', Actual: '{autologging_tag_value}'"
    assert RunStatus.is_terminated(RunStatus.from_string(run.info.status)), f"Autologging run with id {run_id} has a non-terminal status '{run.info.status}'"