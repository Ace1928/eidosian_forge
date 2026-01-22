from datetime import datetime, timezone
import sqlalchemy as sa
from sqlalchemy.types import PickleType
from celery import states
from .session import ResultModelBase
class TaskSet(ResultModelBase):
    """TaskSet result."""
    __tablename__ = 'celery_tasksetmeta'
    __table_args__ = {'sqlite_autoincrement': True}
    id = sa.Column(sa.Integer, sa.Sequence('taskset_id_sequence'), autoincrement=True, primary_key=True)
    taskset_id = sa.Column(sa.String(155), unique=True)
    result = sa.Column(PickleType, nullable=True)
    date_done = sa.Column(sa.DateTime, default=datetime.now(timezone.utc), nullable=True)

    def __init__(self, taskset_id, result):
        self.taskset_id = taskset_id
        self.result = result

    def to_dict(self):
        return {'taskset_id': self.taskset_id, 'result': self.result, 'date_done': self.date_done}

    def __repr__(self):
        return f'<TaskSet: {self.taskset_id}>'

    @classmethod
    def configure(cls, schema=None, name=None):
        cls.__table__.schema = schema
        cls.id.default.schema = schema
        cls.__table__.name = name or cls.__tablename__