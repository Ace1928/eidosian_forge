import uuid
from oslo_db.sqlalchemy import models
from oslo_serialization import jsonutils
from sqlalchemy import BigInteger
from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import Enum
from sqlalchemy import ForeignKey
from sqlalchemy import Index
from sqlalchemy import Integer
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import backref, relationship
from sqlalchemy import sql
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy.types import TypeDecorator
from sqlalchemy import UniqueConstraint
from glance.common import timeutils
class ImageLocation(BASE, GlanceBase):
    """Represents an image location in the datastore."""
    __tablename__ = 'image_locations'
    __table_args__ = (Index('ix_image_locations_image_id', 'image_id'), Index('ix_image_locations_deleted', 'deleted'))
    id = Column(Integer, primary_key=True, nullable=False)
    image_id = Column(String(36), ForeignKey('images.id'), nullable=False)
    image = relationship(Image, backref=backref('locations'))
    value = Column(Text(), nullable=False)
    meta_data = Column(JSONEncodedDict(), default={})
    status = Column(String(30), server_default='active', nullable=False)