from oslo_utils import timeutils
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy.orm import object_mapper
from oslo_db.sqlalchemy import types
class SoftDeleteMixin(object):
    deleted_at = Column(DateTime)
    deleted = Column(types.SoftDeleteInteger, default=0)

    def soft_delete(self, session):
        """Mark this object as deleted."""
        self.deleted = self.id
        self.deleted_at = timeutils.utcnow()
        self.save(session=session)