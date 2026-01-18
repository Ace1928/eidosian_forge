from typing import List, Optional
from mlflow.entities.model_registry import (
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_uc_registry_messages_pb2 import ModelVersion as ProtoModelVersion
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
from mlflow.protos.databricks_uc_registry_messages_pb2 import TemporaryCredentials
from mlflow.store.artifact.artifact_repo import ArtifactRepository
def uc_model_version_status_to_string(status):
    return _STATUS_TO_STRING[status]