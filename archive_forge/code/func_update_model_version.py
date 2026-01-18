import logging
from mlflow.entities.model_registry import ModelVersionTag, RegisteredModelTag
from mlflow.exceptions import MlflowException
from mlflow.store.model_registry import (
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS, utils
from mlflow.utils.arguments_utils import _get_arg_names
def update_model_version(self, name, version, description):
    """Update metadata associated with a model version in backend.

        Args:
            name: Name of the containing registered model.
            version: Version number of the model version.
            description: New description.
        """
    return self.store.update_model_version(name=name, version=version, description=description)