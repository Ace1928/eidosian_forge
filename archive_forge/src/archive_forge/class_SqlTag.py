import time
from sqlalchemy import (
from sqlalchemy.orm import backref, declarative_base, relationship
class SqlTag(Base):
    """
    DB model for :py:class:`mlflow.entities.RunTag`. These are recorded in ``tags`` table.
    """
    __tablename__ = 'tags'
    key = Column(String(250))
    '\n    Tag key: `String` (limit 250 characters). *Primary Key* for ``tags`` table.\n    '
    value = Column(String(250), nullable=True)
    '\n    Value associated with tag: `String` (limit 250 characters). Could be *null*.\n    '
    run_uuid = Column(String(32), ForeignKey('runs.run_uuid'))
    '\n    Run UUID to which this tag belongs to: *Foreign Key* into ``runs`` table.\n    '
    run = relationship('SqlRun', backref=backref('tags', cascade='all'))
    '\n    SQLAlchemy relationship (many:one) with :py:class:`mlflow.store.dbmodels.models.SqlRun`.\n    '
    __table_args__ = (PrimaryKeyConstraint('key', 'run_uuid', name='tag_pk'),)

    def __repr__(self):
        return f'<SqlRunTag({self.key}, {self.value})>'