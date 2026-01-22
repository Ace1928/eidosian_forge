import logging
import os
import posixpath
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.utils.autologging_utils import (
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import LATEST_CHECKPOINT_ARTIFACT_TAG_KEY
def _is_new_checkpoint_better(self, new_monitor_value):
    if self.last_monitor_value is None:
        return True
    if self.mode == 'min':
        return new_monitor_value < self.last_monitor_value
    return new_monitor_value > self.last_monitor_value