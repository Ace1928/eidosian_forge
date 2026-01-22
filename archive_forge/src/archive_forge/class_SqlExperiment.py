import time
from sqlalchemy import (
from sqlalchemy.orm import backref, declarative_base, relationship
class SqlExperiment(Base):
    """
    DB model for :py:class:`mlflow.entities.Experiment`. These are recorded in ``experiment`` table.
    """
    __tablename__ = 'experiments'
    experiment_id = Column(Integer, autoincrement=True)
    '\n    Experiment ID: `Integer`. *Primary Key* for ``experiment`` table.\n    '
    name = Column(String(256), unique=True, nullable=False)
    '\n    Experiment name: `String` (limit 256 characters). Defined as *Unique* and *Non null* in\n                     table schema.\n    '
    artifact_location = Column(String(256), nullable=True)
    '\n    Default artifact location for this experiment: `String` (limit 256 characters). Defined as\n                                                    *Non null* in table schema.\n    '
    lifecycle_stage = Column(String(32), default='active')
    '\n    Lifecycle Stage of experiment: `String` (limit 32 characters).\n                                    Can be either ``active`` (default) or ``deleted``.\n    '
    __table_args__ = (CheckConstraint(lifecycle_stage.in_(['active', 'deleted']), name='experiments_lifecycle_stage'), PrimaryKeyConstraint('experiment_id', name='experiment_pk'))

    def __repr__(self):
        return f'<SqlExperiment ({self.experiment_id}, {self.name})>'