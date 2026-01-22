import logging
import os
import tempfile
import warnings
from packaging.version import Version
import mlflow.pytorch
from mlflow.exceptions import MlflowException
from mlflow.ml_package_versions import _ML_PACKAGE_VERSIONS
from mlflow.pytorch import _pytorch_autolog
from mlflow.utils.autologging_utils import (
from mlflow.utils.checkpoint_utils import MlflowModelCheckpointCallbackBase
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
def _log_early_stop_metrics(early_stop_callback, client, run_id):
    """
    Logs early stopping behavior results (e.g. stopped epoch) as metrics to MLflow.

    Args:
        early_stop_callback: The early stopping callback instance used during training.
        client: An `MlflowAutologgingQueueingClient` instance used for MLflow logging.
        run_id: The ID of the MLflow Run to which to log configuration parameters.
    """
    if early_stop_callback.stopped_epoch == 0:
        return
    metrics = {'stopped_epoch': early_stop_callback.stopped_epoch, 'restored_epoch': early_stop_callback.stopped_epoch - max(1, early_stop_callback.patience)}
    if hasattr(early_stop_callback, 'best_score'):
        metrics['best_score'] = float(early_stop_callback.best_score)
    if hasattr(early_stop_callback, 'wait_count'):
        metrics['wait_count'] = early_stop_callback.wait_count
    client.log_metrics(run_id, metrics)