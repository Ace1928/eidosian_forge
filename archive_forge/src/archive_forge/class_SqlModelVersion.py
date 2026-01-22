from sqlalchemy import (
from sqlalchemy.orm import backref, relationship
from mlflow.entities.model_registry import (
from mlflow.entities.model_registry.model_version_stages import STAGE_DELETED_INTERNAL, STAGE_NONE
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.store.db.base_sql_model import Base
from mlflow.utils.time import get_current_time_millis
class SqlModelVersion(Base):
    __tablename__ = 'model_versions'
    name = Column(String(256), ForeignKey('registered_models.name', onupdate='cascade'))
    version = Column(Integer, nullable=False)
    creation_time = Column(BigInteger, default=get_current_time_millis)
    last_updated_time = Column(BigInteger, nullable=True, default=None)
    description = Column(String(5000), nullable=True)
    user_id = Column(String(256), nullable=True, default=None)
    current_stage = Column(String(20), default=STAGE_NONE)
    source = Column(String(500), nullable=True, default=None)
    storage_location = Column(String(500), nullable=True, default=None)
    run_id = Column(String(32), nullable=True, default=None)
    run_link = Column(String(500), nullable=True, default=None)
    status = Column(String(20), default=ModelVersionStatus.to_string(ModelVersionStatus.READY))
    status_message = Column(String(500), nullable=True, default=None)
    registered_model = relationship('SqlRegisteredModel', backref=backref('model_versions', cascade='all'))
    __table_args__ = (PrimaryKeyConstraint('name', 'version', name='model_version_pk'),)

    def to_mlflow_entity(self):
        return ModelVersion(self.name, self.version, self.creation_time, self.last_updated_time, self.description, self.user_id, self.current_stage, self.source, self.run_id, self.status, self.status_message, [tag.to_mlflow_entity() for tag in self.model_version_tags], self.run_link, [])