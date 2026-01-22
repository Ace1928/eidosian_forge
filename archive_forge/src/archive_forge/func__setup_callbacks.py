import importlib
import logging
import os
import shutil
import tempfile
from typing import Any, Dict, NamedTuple, Optional
import numpy as np
import pandas
import yaml
from packaging.version import Version
import mlflow
from mlflow import pyfunc
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.numpy_dataset import from_numpy
from mlflow.data.tensorflow_dataset import from_tensorflow
from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature, infer_signature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.tensorflow.callback import MlflowCallback, MlflowModelCheckpointCallback  # noqa: F401
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking.context import registry as context_registry
from mlflow.types.schema import TensorSpec
from mlflow.utils import is_iterator
from mlflow.utils.autologging_utils import (
from mlflow.utils.checkpoint_utils import (
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import TempDir, get_total_file_size, write_to
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
def _setup_callbacks(callbacks, log_every_epoch, log_every_n_steps):
    """
    Adds TensorBoard and MlfLowTfKeras callbacks to the
    input list, and returns the new list and appropriate log directory.
    """
    from mlflow.tensorflow.autologging import _TensorBoard
    from mlflow.tensorflow.callback import MlflowCallback, MlflowModelCheckpointCallback
    tb = _get_tensorboard_callback(callbacks)
    for callback in callbacks:
        if isinstance(callback, MlflowCallback):
            raise MlflowException('MLflow autologging must be turned off if an `MlflowCallback` is explicitly added to the callback list. You are creating an `MlflowCallback` while having autologging enabled. Please either call `mlflow.tensorflow.autolog(disable=True)` to disable autologging or remove `MlflowCallback` from the callback list. ')
    if tb is None:
        log_dir = _TensorBoardLogDir(location=tempfile.mkdtemp(), is_temp=True)
        callbacks.append(_TensorBoard(log_dir.location))
    else:
        log_dir = _TensorBoardLogDir(location=tb.log_dir, is_temp=False)
    callbacks.append(MlflowCallback(log_every_epoch=log_every_epoch, log_every_n_steps=log_every_n_steps))
    model_checkpoint = get_autologging_config(mlflow.tensorflow.FLAVOR_NAME, 'checkpoint', True)
    if model_checkpoint:
        checkpoint_monitor = get_autologging_config(mlflow.tensorflow.FLAVOR_NAME, 'checkpoint_monitor', 'val_loss')
        checkpoint_mode = get_autologging_config(mlflow.tensorflow.FLAVOR_NAME, 'checkpoint_mode', 'min')
        checkpoint_save_best_only = get_autologging_config(mlflow.tensorflow.FLAVOR_NAME, 'checkpoint_save_best_only', True)
        checkpoint_save_weights_only = get_autologging_config(mlflow.tensorflow.FLAVOR_NAME, 'checkpoint_save_weights_only', False)
        checkpoint_save_freq = get_autologging_config(mlflow.tensorflow.FLAVOR_NAME, 'checkpoint_save_freq', 'epoch')
        if not any((isinstance(callback, MlflowModelCheckpointCallback) for callback in callbacks)):
            callbacks.append(MlflowModelCheckpointCallback(monitor=checkpoint_monitor, mode=checkpoint_mode, save_best_only=checkpoint_save_best_only, save_weights_only=checkpoint_save_weights_only, save_freq=checkpoint_save_freq))
    return (callbacks, log_dir)