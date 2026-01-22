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
class FitPatch(PatchFunction):

    def __init__(self):
        self.log_dir = None

    def _patch_implementation(self, original, inst, *args, **kwargs):
        unlogged_params = ['self', 'x', 'y', 'callbacks', 'validation_data', 'verbose']
        batch_size = None
        try:
            is_single_input_model = isinstance(inst.input_shape, tuple)
            training_data = kwargs['x'] if 'x' in kwargs else args[0]
            if isinstance(training_data, tf.data.Dataset) and hasattr(training_data, '_batch_size'):
                batch_size = training_data._batch_size.numpy()
            elif isinstance(training_data, tf.keras.utils.Sequence):
                first_batch_inputs, *_ = training_data[0]
                if is_single_input_model:
                    batch_size = len(first_batch_inputs)
                else:
                    batch_size = len(first_batch_inputs[0])
            elif is_iterator(training_data):
                peek = next(training_data)
                batch_size = len(peek[0]) if is_single_input_model else len(peek[0][0])

                def __restore_generator(prev_generator):
                    yield peek
                    yield from prev_generator
                restored_generator = __restore_generator(training_data)
                if 'x' in kwargs:
                    kwargs['x'] = restored_generator
                else:
                    args = (restored_generator,) + args[1:]
        except Exception as e:
            _logger.warning('Encountered unexpected error while inferring batch size from training dataset: %s', e)
        if batch_size is not None:
            mlflow.log_param('batch_size', batch_size)
            unlogged_params.append('batch_size')
        log_fn_args_as_params(original, args, kwargs, unlogged_params)
        if len(args) >= 6:
            args = list(args)
            callbacks = list(args[5])
            callbacks, self.log_dir = _setup_callbacks(callbacks, log_every_epoch=log_every_epoch, log_every_n_steps=log_every_n_steps)
            args[5] = callbacks
            args = tuple(args)
        else:
            callbacks = list(kwargs.get('callbacks') or [])
            kwargs['callbacks'], self.log_dir = _setup_callbacks(callbacks, log_every_epoch=log_every_epoch, log_every_n_steps=log_every_n_steps)
        early_stop_callback = _get_early_stop_callback(callbacks)
        _log_early_stop_callback_params(early_stop_callback)
        if log_datasets:
            try:
                context_tags = context_registry.resolve_tags()
                source = CodeDatasetSource(tags=context_tags)
                x = kwargs['x'] if 'x' in kwargs else args[0]
                if 'y' in kwargs:
                    y = kwargs['y']
                elif len(args) >= 2:
                    y = args[1]
                else:
                    y = None
                if 'validation_data' in kwargs:
                    validation_data = kwargs['validation_data']
                elif len(args) >= 8:
                    validation_data = args[7]
                else:
                    validation_data = None
                _log_tensorflow_dataset(x, source, 'train', targets=y)
                if validation_data is not None:
                    _log_tensorflow_dataset(validation_data, source, 'eval')
            except Exception as e:
                _logger.warning('Failed to log training dataset information to MLflow Tracking. Reason: %s', e)
        history = original(inst, *args, **kwargs)
        if log_models:
            _log_keras_model(history, args)
        _log_early_stop_callback_metrics(callback=early_stop_callback, history=history)
        mlflow.flush_async_logging()
        mlflow.log_artifacts(local_dir=self.log_dir.location, artifact_path='tensorboard_logs')
        if self.log_dir.is_temp:
            shutil.rmtree(self.log_dir.location)
        return history

    def _on_exception(self, exception):
        if self.log_dir is not None and self.log_dir.is_temp and os.path.exists(self.log_dir.location):
            shutil.rmtree(self.log_dir.location)