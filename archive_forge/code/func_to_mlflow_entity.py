from sqlalchemy import (
from sqlalchemy.orm import backref, relationship
from mlflow.entities.model_registry import (
from mlflow.entities.model_registry.model_version_stages import STAGE_DELETED_INTERNAL, STAGE_NONE
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.store.db.base_sql_model import Base
from mlflow.utils.time import get_current_time_millis
def to_mlflow_entity(self):
    return RegisteredModelAlias(self.alias, self.version)