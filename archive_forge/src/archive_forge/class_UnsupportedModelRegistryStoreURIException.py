import warnings
from abc import ABCMeta, abstractmethod
import entrypoints
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.uri import get_uri_scheme
class UnsupportedModelRegistryStoreURIException(MlflowException):
    """Exception thrown when building a model registry store with an unsupported URI"""

    def __init__(self, unsupported_uri, supported_uri_schemes):
        message = f" Model registry functionality is unavailable; got unsupported URI '{unsupported_uri}' for model registry data storage. Supported URI schemes are: {supported_uri_schemes}. See https://www.mlflow.org/docs/latest/tracking.html#storage for how to run an MLflow server against one of the supported backend storage locations."
        super().__init__(message, error_code=INVALID_PARAMETER_VALUE)
        self.supported_uri_schemes = supported_uri_schemes