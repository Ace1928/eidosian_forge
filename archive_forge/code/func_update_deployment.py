import logging
import os
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.openai_utils import (
from mlflow.utils.rest_utils import augmented_raise_for_status
from mlflow.deployments import BaseDeploymentClient
def update_deployment(self, name, model_uri=None, flavor=None, config=None, endpoint=None):
    """
        .. warning::

            This method is not implemented for `OpenAIDeploymentClient`.
        """
    raise NotImplementedError