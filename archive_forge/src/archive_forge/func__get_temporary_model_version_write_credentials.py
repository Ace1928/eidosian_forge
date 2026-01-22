import base64
import functools
import logging
import os
import shutil
from contextlib import contextmanager
import mlflow
from mlflow.entities import Run
from mlflow.environment_variables import MLFLOW_UNITY_CATALOG_PRESIGNED_URLS_ENABLED
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
from mlflow.protos.databricks_uc_registry_service_pb2 import UcModelRegistryService
from mlflow.protos.service_pb2 import GetRun, MlflowService
from mlflow.store.artifact.presigned_url_artifact_repo import PresignedUrlArtifactRepository
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.model_registry.rest_store import BaseRestStore
from mlflow.utils._spark_utils import _get_active_spark_session
from mlflow.utils._unity_catalog_utils import (
from mlflow.utils.annotations import experimental
from mlflow.utils.databricks_utils import get_databricks_host_creds, is_databricks_uri
from mlflow.utils.mlflow_tags import (
from mlflow.utils.proto_json_utils import message_to_json, parse_dict
from mlflow.utils.rest_utils import (
def _get_temporary_model_version_write_credentials(self, name, version) -> TemporaryCredentials:
    """
        Get temporary credentials for uploading model version files

        Args:
            name: Registered model name.
            version: Model version number.

        Returns:
            mlflow.protos.databricks_uc_registry_messages_pb2.TemporaryCredentials containing
            temporary model version credentials.
        """
    req_body = message_to_json(GenerateTemporaryModelVersionCredentialsRequest(name=name, version=version, operation=MODEL_VERSION_OPERATION_READ_WRITE))
    return self._call_endpoint(GenerateTemporaryModelVersionCredentialsRequest, req_body).credentials