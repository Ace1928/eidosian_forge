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
class _SHAPWrapper:

    def __init__(self, path):
        flavor_conf = _get_flavor_configuration(model_path=path, flavor_name=FLAVOR_NAME)
        shap_explainer_artifacts_path = os.path.join(path, flavor_conf['serialized_explainer'])
        underlying_model_flavor = flavor_conf['underlying_model_flavor']
        model = None
        if underlying_model_flavor != _UNKNOWN_MODEL_FLAVOR:
            underlying_model_path = os.path.join(path, _UNDERLYING_MODEL_SUBPATH)
            if underlying_model_flavor == mlflow.sklearn.FLAVOR_NAME:
                model = mlflow.sklearn._load_pyfunc(underlying_model_path).predict
            elif underlying_model_flavor == mlflow.pytorch.FLAVOR_NAME:
                model = mlflow.pytorch._load_model(os.path.join(underlying_model_path, 'data'))
        self.explainer = _load_explainer(explainer_file=shap_explainer_artifacts_path, model=model)

    def predict(self, dataframe, params: Optional[Dict[str, Any]]=None):
        """
        Args:
            dataframe: Model input data.
            params: Additional parameters to pass to the model for inference.

                .. Note:: Experimental: This parameter may change or be removed in a future
                    release without warning.

        Returns:
            Model predictions.
        """
        return self.explainer(dataframe.values).values