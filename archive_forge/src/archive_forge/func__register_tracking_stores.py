import logging
import os
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Union
from mlflow.environment_variables import MLFLOW_TRACKING_URI
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.store.tracking import DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH
from mlflow.store.tracking.file_store import FileStore
from mlflow.store.tracking.rest_store import RestStore
from mlflow.tracking._tracking_service.registry import TrackingStoreRegistry
from mlflow.utils.credentials import get_default_host_creds
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.file_utils import path_to_local_file_uri
from mlflow.utils.uri import _DATABRICKS_UNITY_CATALOG_SCHEME
def _register_tracking_stores():
    global _tracking_store_registry
    _tracking_store_registry.register('', _get_file_store)
    _tracking_store_registry.register('file', _get_file_store)
    _tracking_store_registry.register('databricks', _get_databricks_rest_store)
    _tracking_store_registry.register(_DATABRICKS_UNITY_CATALOG_SCHEME, _get_databricks_uc_rest_store)
    for scheme in ['http', 'https']:
        _tracking_store_registry.register(scheme, _get_rest_store)
    for scheme in DATABASE_ENGINES:
        _tracking_store_registry.register(scheme, _get_sqlalchemy_store)
    _tracking_store_registry.register_entrypoints()