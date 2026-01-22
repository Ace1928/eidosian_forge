import sqlalchemy as sa
from sqlalchemy import (
from sqlalchemy.orm import backref, relationship
from mlflow.entities import (
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.store.db.base_sql_model import Base
from mlflow.utils.mlflow_tags import _get_run_name_from_tags
from mlflow.utils.time import get_current_time_millis
class SqlInput(Base):
    __tablename__ = 'inputs'
    __table_args__ = (PrimaryKeyConstraint('source_type', 'source_id', 'destination_type', 'destination_id', name='inputs_pk'), Index(f'index_{__tablename__}_input_uuid', 'input_uuid'), Index(f'index_{__tablename__}_destination_type_destination_id_source_type', 'destination_type', 'destination_id', 'source_type'))
    input_uuid = Column(String(36), nullable=False)
    '\n    Input UUID: `String` (limit 36 characters). Defined as *Non-null* in schema.\n    '
    source_type = Column(String(36), nullable=False)
    '\n    Source type: `String` (limit 36 characters). Defined as *Non-null* in schema.\n    Part of *Primary Key* for ``inputs`` table.\n    '
    source_id = Column(String(36), nullable=False)
    '\n    Source Id: `String` (limit 36 characters). Defined as *Non-null* in schema.\n    Part of *Primary Key* for ``inputs`` table.\n    '
    destination_type = Column(String(36), nullable=False)
    '\n    Destination type: `String` (limit 36 characters). Defined as *Non-null* in schema.\n    Part of *Primary Key* for ``inputs`` table.\n    '
    destination_id = Column(String(36), nullable=False)
    '\n    Destination Id: `String` (limit 36 characters). Defined as *Non-null* in schema.\n    Part of *Primary Key* for ``inputs`` table.\n    '

    def __repr__(self):
        return '<SqlInput ({}, {}, {}, {}, {})>'.format(self.input_uuid, self.source_type, self.source_id, self.destination_type, self.destination_id)