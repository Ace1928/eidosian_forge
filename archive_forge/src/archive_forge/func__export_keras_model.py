import importlib
import logging
import os
import shutil
import tempfile
import keras
import yaml
import mlflow
from mlflow import pyfunc
from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
from mlflow.models import (
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.types.schema import TensorSpec
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import get_total_file_size, write_to
from mlflow.utils.requirements_utils import _get_pinned_requirement
def _export_keras_model(model, path, signature):
    if signature is None:
        raise ValueError('`signature` cannot be None when `save_exported_model=True` for `mlflow.keras.save_model()` method.')
    try:
        import tensorflow as tf
    except ImportError:
        raise MlflowException('`tensorflow` must be installed if you want to export a Keras 3 model, please install `tensorflow` by `pip install tensorflow`, or set `save_exported_model=False`.')
    input_schema = signature.inputs.to_dict()
    export_signature = []
    for schema in input_schema:
        dtype = schema['tensor-spec']['dtype']
        shape = schema['tensor-spec']['shape']
        new_shape = [size if size != -1 else None for size in shape]
        export_signature.append(tf.TensorSpec(shape=new_shape, dtype=dtype))
    export_archive = keras.export.ExportArchive()
    export_archive.track(model)
    export_archive.add_endpoint(name='serve', fn=model.call, input_signature=export_signature)
    export_archive.write_out(path)