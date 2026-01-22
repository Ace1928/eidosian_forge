import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from threading import get_ident as get_current_thread_id
import mlflow
from mlflow.utils import logging_utils
def _patched_showwarning(self, message, category, filename, lineno, *args, **kwargs):
    """
        A patched implementation of `warnings.showwarning` that enforces the warning configuration
        options configured on the controller (e.g. rerouting or disablement of MLflow warnings,
        disablement of all warnings for the current thread).

        Note that reassigning `warnings.showwarning` is the standard / recommended approach for
        modifying warning message display behaviors. For reference, see
        https://docs.python.org/3/library/warnings.html#warnings.showwarning
        """
    from mlflow.utils.autologging_utils import _logger
    warning_source_path = Path(filename).resolve()
    is_mlflow_warning = self._mlflow_root_path in warning_source_path.parents
    curr_thread = get_current_thread_id()
    if curr_thread in self._disabled_threads or (is_mlflow_warning and self._mlflow_warnings_disabled_globally):
        return
    elif curr_thread in self._rerouted_threads and (not is_mlflow_warning) or (is_mlflow_warning and self._mlflow_warnings_rerouted_to_event_logs):
        _logger.warning('MLflow autologging encountered a warning: "%s:%d: %s: %s"', filename, lineno, category.__name__, message)
    else:
        self._original_showwarning(message, category, filename, lineno, *args, **kwargs)