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
def _save_keras_custom_objects(path, custom_objects, file_name):
    """
    Save custom objects dictionary to a cloudpickle file so a model can be easily loaded later.

    Args:
        path: An absolute path that points to the data directory within /path/to/model.
        custom_objects: Keras ``custom_objects`` is a dictionary mapping
            names (strings) to custom classes or functions to be considered
            during deserialization. MLflow saves these custom layers using
            CloudPickle and restores them automatically when the model is
            loaded with :py:func:`mlflow.keras.load_model` and
            :py:func:`mlflow.pyfunc.load_model`.
        file_name: The file name to save the custom objects to.
    """
    import cloudpickle
    custom_objects_path = os.path.join(path, file_name)
    with open(custom_objects_path, 'wb') as out_f:
        cloudpickle.dump(custom_objects, out_f)