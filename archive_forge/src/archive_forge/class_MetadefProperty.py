from oslo_db.sqlalchemy import models
from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy import Index
from sqlalchemy import Integer
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy import UniqueConstraint
from glance.common import timeutils
from glance.db.sqlalchemy.models import JSONEncodedDict
class MetadefProperty(BASE_DICT, GlanceMetadefBase):
    """Represents a metadata-schema namespace-property in the datastore."""
    __tablename__ = 'metadef_properties'
    __table_args__ = (UniqueConstraint('namespace_id', 'name', name='uq_metadef_properties_namespace_id_name'), Index('ix_metadef_properties_name', 'name'))
    id = Column(Integer, primary_key=True, nullable=False)
    namespace_id = Column(Integer(), ForeignKey('metadef_namespaces.id'), nullable=False)
    name = Column(String(80), nullable=False)
    json_schema = Column(JSONEncodedDict(), default={}, nullable=False)