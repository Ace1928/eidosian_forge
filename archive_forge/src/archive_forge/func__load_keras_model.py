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
def _load_keras_model(model_path, keras_module, save_format, **kwargs):
    keras_models = importlib.import_module(keras_module.__name__ + '.models')
    custom_objects = kwargs.pop('custom_objects', {})
    if (saved_custom_objects := _load_custom_objects(model_path, _CUSTOM_OBJECTS_SAVE_PATH)):
        saved_custom_objects.update(custom_objects)
        custom_objects = saved_custom_objects
    if (global_custom_objects := _load_custom_objects(model_path, _GLOBAL_CUSTOM_OBJECTS_SAVE_PATH)):
        global_custom_objects.update(custom_objects)
        custom_objects = global_custom_objects
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, _MODEL_SAVE_PATH)
    if save_format == 'h5':
        model_path += '.h5'
    elif os.path.exists(model_path + '.keras'):
        model_path += '.keras'
    import tensorflow as tf
    if save_format == 'h5' and (2, 2, 3) <= Version(tf.__version__).release < (2, 16):
        import h5py
        with h5py.File(os.path.abspath(model_path), 'r') as model_path:
            return keras_models.load_model(model_path, custom_objects=custom_objects, **kwargs)
    else:
        return keras_models.load_model(model_path, custom_objects=custom_objects, **kwargs)