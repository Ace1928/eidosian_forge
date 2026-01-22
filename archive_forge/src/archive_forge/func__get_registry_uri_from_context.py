from functools import partial
from mlflow.environment_variables import MLFLOW_REGISTRY_URI
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.store.model_registry.databricks_workspace_model_registry_rest_store import (
from mlflow.store.model_registry.file_store import FileStore
from mlflow.store.model_registry.rest_store import RestStore
from mlflow.tracking._model_registry.registry import ModelRegistryStoreRegistry
from mlflow.tracking._tracking_service.utils import (
from mlflow.utils._spark_utils import _get_active_spark_session
from mlflow.utils.credentials import get_default_host_creds
from mlflow.utils.databricks_utils import (
from mlflow.utils.uri import _DATABRICKS_UNITY_CATALOG_SCHEME
def _get_registry_uri_from_context():
    global _registry_uri
    if _registry_uri is not None:
        return _registry_uri
    elif (uri := MLFLOW_REGISTRY_URI.get()) or (uri := _get_registry_uri_from_spark_session()):
        return uri
    return _registry_uri