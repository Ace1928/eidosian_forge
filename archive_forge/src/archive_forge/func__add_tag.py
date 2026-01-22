from mlflow.entities.model_registry._model_registry_entity import _ModelRegistryEntity
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.entities.model_registry.model_version_tag import ModelVersionTag
from mlflow.protos.model_registry_pb2 import ModelVersion as ProtoModelVersion
from mlflow.protos.model_registry_pb2 import ModelVersionTag as ProtoModelVersionTag
def _add_tag(self, tag):
    self._tags[tag.key] = tag.value