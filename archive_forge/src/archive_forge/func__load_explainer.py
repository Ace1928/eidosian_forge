import os
import tempfile
import types
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Optional
import numpy as np
import yaml
import mlflow
import mlflow.utils.autologging_utils
from mlflow import pyfunc
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_package_name
from mlflow.utils.uri import append_to_uri_path
def _load_explainer(explainer_file, model=None):
    """
    Load a SHAP explainer saved as an MLflow artifact on the local file system.

    Args:
        explainer_file: Local filesystem path to the MLflow Model saved with the ``shap`` flavor.
        model: Model to override underlying explainer model.

    """
    import shap

    def inject_model_loader(_in_file):
        return model
    with open(explainer_file, 'rb') as explainer:
        if model is None:
            explainer = shap.Explainer.load(explainer)
        else:
            explainer = shap.Explainer.load(explainer, model_loader=inject_model_loader)
        return explainer