import sqlalchemy as sa
from sqlalchemy import (
from sqlalchemy.orm import backref, relationship
from mlflow.entities import (
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.store.db.base_sql_model import Base
from mlflow.utils.mlflow_tags import _get_run_name_from_tags
from mlflow.utils.time import get_current_time_millis
class SqlLatestMetric(Base):
    __tablename__ = 'latest_metrics'
    __table_args__ = (PrimaryKeyConstraint('key', 'run_uuid', name='latest_metric_pk'), Index(f'index_{__tablename__}_run_uuid', 'run_uuid'))
    key = Column(String(250))
    '\n    Metric key: `String` (limit 250 characters). Part of *Primary Key* for ``latest_metrics`` table.\n    '
    value = Column(sa.types.Float(precision=53), nullable=False)
    '\n    Metric value: `Float`. Defined as *Non-null* in schema.\n    '
    timestamp = Column(BigInteger, default=get_current_time_millis)
    '\n    Timestamp recorded for this metric entry: `BigInteger`. Part of *Primary Key* for\n                                               ``latest_metrics`` table.\n    '
    step = Column(BigInteger, default=0, nullable=False)
    '\n    Step recorded for this metric entry: `BigInteger`.\n    '
    is_nan = Column(Boolean(create_constraint=True), nullable=False, default=False)
    '\n    True if the value is in fact NaN.\n    '
    run_uuid = Column(String(32), ForeignKey('runs.run_uuid'))
    '\n    Run UUID to which this metric belongs to: Part of *Primary Key* for ``latest_metrics`` table.\n                                              *Foreign Key* into ``runs`` table.\n    '
    run = relationship('SqlRun', backref=backref('latest_metrics', cascade='all'))
    '\n    SQLAlchemy relationship (many:one) with :py:class:`mlflow.store.dbmodels.models.SqlRun`.\n    '

    def __repr__(self):
        return f'<SqlLatestMetric({self.key}, {self.value}, {self.timestamp}, {self.step})>'

    def to_mlflow_entity(self):
        """
        Convert DB model to corresponding MLflow entity.

        Returns:
            mlflow.entities.Metric: Description of the return value.
        """
        return Metric(key=self.key, value=self.value if not self.is_nan else float('nan'), timestamp=self.timestamp, step=self.step)