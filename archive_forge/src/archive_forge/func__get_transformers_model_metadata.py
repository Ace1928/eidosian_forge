import json
import logging
import pathlib
import re
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import yaml
from packaging.version import Version
import mlflow
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature, infer_pip_requirements
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.types.llm import (
from mlflow.types.schema import ColSpec, Schema, TensorSpec
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import (
from mlflow.utils.environment import (
from mlflow.utils.file_utils import get_total_file_size, write_to
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
def _get_transformers_model_metadata(model) -> Dict[str, str]:
    """
    Extract metadata about the underlying Transformers model, such as the model class name
    and the repository id.

    Args:
        model: A SentenceTransformer model instance.

    Returns:
        A dictionary containing metadata about the Transformers model.
    """
    from sentence_transformers.models import Transformer
    for module in model.modules():
        if isinstance(module, Transformer):
            model_instance = module.auto_model
            return {_TRANSFORMER_SOURCE_MODEL_NAME_KEY: _get_transformers_model_name(model_instance.name_or_path), _TRANSFORMER_MODEL_TYPE_KEY: model_instance.__class__.__name__}
    return {}