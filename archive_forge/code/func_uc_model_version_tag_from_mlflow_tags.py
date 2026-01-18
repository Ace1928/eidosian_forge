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
def uc_model_version_tag_from_mlflow_tags(tags: Optional[List[ModelVersionTag]]) -> List[ProtoModelVersionTag]:
    if tags is None:
        return []
    return [ProtoModelVersionTag(key=t.key, value=t.value) for t in tags]