import logging
from mlflow.entities.model_registry import ModelVersionTag, RegisteredModelTag
from mlflow.exceptions import MlflowException
from mlflow.store.model_registry import (
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS, utils
from mlflow.utils.arguments_utils import _get_arg_names
def update_registered_model(self, name, description):
    """Updates description for RegisteredModel entity.

        Backend raises exception if a registered model with given name does not exist.

        Args:
            name: Name of the registered model to update.
            description: New description.

        Returns:
            A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.

        """
    return self.store.update_registered_model(name=name, description=description)