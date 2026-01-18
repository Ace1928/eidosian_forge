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
def with_managed_run(autologging_integration, patch_function, tags=None):
    """Given a `patch_function`, returns an `augmented_patch_function` that wraps the execution of
    `patch_function` with an active MLflow run. The following properties apply:

        - An MLflow run is only created if there is no active run present when the
          patch function is executed

        - If an active run is created by the `augmented_patch_function`, it is terminated
          with the `FINISHED` state at the end of function execution

        - If an active run is created by the `augmented_patch_function`, it is terminated
          with the `FAILED` if an unhandled exception is thrown during function execution

    Note that, if nested runs or non-fluent runs are created by `patch_function`, `patch_function`
    is responsible for terminating them by the time it terminates
    (or in the event of an exception).

    Args:
        autologging_integration: The autologging integration associated
            with the `patch_function`.
        patch_function: A `PatchFunction` class definition or a function object
            compatible with `safe_patch`.
        tags: A dictionary of string tags to set on each managed run created during the
            execution of `patch_function`.
    """

    def create_managed_run():
        managed_run = mlflow.start_run(tags=tags)
        _logger.info("Created MLflow autologging run with ID '%s', which will track hyperparameters, performance metrics, model artifacts, and lineage information for the current %s workflow", managed_run.info.run_id, autologging_integration)
        return managed_run
    if inspect.isclass(patch_function):

        class PatchWithManagedRun(patch_function):

            def __init__(self):
                super().__init__()
                self.managed_run = None

            def _patch_implementation(self, original, *args, **kwargs):
                if not mlflow.active_run():
                    self.managed_run = create_managed_run()
                result = super()._patch_implementation(original, *args, **kwargs)
                if self.managed_run:
                    mlflow.end_run(RunStatus.to_string(RunStatus.FINISHED))
                return result

            def _on_exception(self, e):
                if self.managed_run:
                    mlflow.end_run(RunStatus.to_string(RunStatus.FAILED))
                super()._on_exception(e)
        return PatchWithManagedRun
    else:

        def patch_with_managed_run(original, *args, **kwargs):
            managed_run = None
            if not mlflow.active_run():
                managed_run = create_managed_run()
            try:
                result = patch_function(original, *args, **kwargs)
            except (Exception, KeyboardInterrupt):
                if managed_run:
                    mlflow.end_run(RunStatus.to_string(RunStatus.FAILED))
                raise
            else:
                if managed_run:
                    mlflow.end_run(RunStatus.to_string(RunStatus.FINISHED))
                return result
        return patch_with_managed_run