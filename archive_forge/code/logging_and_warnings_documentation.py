import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from threading import get_ident as get_current_thread_id
import mlflow
from mlflow.utils import logging_utils

        Returns:
            True if non-MLflow warnings are rerouted to an MLflow event logger with level
            WARNING for the current thread. False otherwise.
        