import sqlalchemy as sa
from sqlalchemy import (
from sqlalchemy.orm import backref, relationship
from mlflow.entities import (
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.store.db.base_sql_model import Base
from mlflow.utils.mlflow_tags import _get_run_name_from_tags
from mlflow.utils.time import get_current_time_millis
class SqlDataset(Base):
    __tablename__ = 'datasets'
    __table_args__ = (PrimaryKeyConstraint('experiment_id', 'name', 'digest', name='dataset_pk'), Index(f'index_{__tablename__}_dataset_uuid', 'dataset_uuid'), Index(f'index_{__tablename__}_experiment_id_dataset_source_type', 'experiment_id', 'dataset_source_type'))
    dataset_uuid = Column(String(36), nullable=False)
    '\n    Dataset UUID: `String` (limit 36 characters). Defined as *Non-null* in schema.\n    Part of *Primary Key* for ``datasets`` table.\n    '
    experiment_id = Column(Integer, ForeignKey('experiments.experiment_id'))
    '\n    Experiment ID to which this dataset belongs: *Foreign Key* into ``experiments`` table.\n    '
    name = Column(String(500), nullable=False)
    '\n    Param name: `String` (limit 500 characters). Defined as *Non-null* in schema.\n    Part of *Primary Key* for ``datasets`` table.\n    '
    digest = Column(String(36), nullable=False)
    '\n    Param digest: `String` (limit 500 characters). Defined as *Non-null* in schema.\n    Part of *Primary Key* for ``datasets`` table.\n    '
    dataset_source_type = Column(String(36), nullable=False)
    '\n    Param dataset_source_type: `String` (limit 36 characters). Defined as *Non-null* in schema.\n    '
    dataset_source = Column(UnicodeText, nullable=False)
    '\n    Param dataset_source: `UnicodeText`. Defined as *Non-null* in schema.\n    '
    dataset_schema = Column(UnicodeText, nullable=True)
    '\n    Param dataset_schema: `UnicodeText`.\n    '
    dataset_profile = Column(UnicodeText, nullable=True)
    '\n    Param dataset_profile: `UnicodeText`.\n    '

    def __repr__(self):
        return '<SqlDataset ({}, {}, {}, {}, {}, {}, {}, {})>'.format(self.dataset_uuid, self.experiment_id, self.name, self.digest, self.dataset_source_type, self.dataset_source, self.dataset_schema, self.dataset_profile)

    def to_mlflow_entity(self):
        """
        Convert DB model to corresponding MLflow entity.

        Returns:
            mlflow.entities.Dataset.
        """
        return Dataset(name=self.name, digest=self.digest, source_type=self.dataset_source_type, source=self.dataset_source, schema=self.dataset_schema, profile=self.dataset_profile)