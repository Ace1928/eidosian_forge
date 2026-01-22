from sqlalchemy import (
from sqlalchemy.orm import declarative_base, relationship
from mlflow.server.auth.entities import ExperimentPermission, RegisteredModelPermission, User
class SqlExperimentPermission(Base):
    __tablename__ = 'experiment_permissions'
    id = Column(Integer(), primary_key=True)
    experiment_id = Column(String(255), nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    permission = Column(String(255))
    __table_args__ = (UniqueConstraint('experiment_id', 'user_id', name='unique_experiment_user'),)

    def to_mlflow_entity(self):
        return ExperimentPermission(experiment_id=self.experiment_id, user_id=self.user_id, permission=self.permission)