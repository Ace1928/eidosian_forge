import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from threading import get_ident as get_current_thread_id
import mlflow
from mlflow.utils import logging_utils
def _should_patch_showwarning(self):
    return len(self._disabled_threads) > 0 or len(self._rerouted_threads) > 0 or self._mlflow_warnings_disabled_globally or self._mlflow_warnings_rerouted_to_event_logs