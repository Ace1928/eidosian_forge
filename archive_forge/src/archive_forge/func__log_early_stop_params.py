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
def _log_early_stop_params(early_stop_callback, client, run_id):
    """
    Logs early stopping configuration parameters to MLflow.

    Args:
        early_stop_callback: The early stopping callback instance used during training.
        client: An `MlflowAutologgingQueueingClient` instance used for MLflow logging.
        run_id: The ID of the MLflow Run to which to log configuration parameters.
    """
    client.log_params(run_id, {p: getattr(early_stop_callback, p) for p in ['monitor', 'mode', 'patience', 'min_delta', 'stopped_epoch'] if hasattr(early_stop_callback, p)})