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
class MetadefNamespaceResourceType(BASE_DICT, GlanceMetadefBase):
    """Represents a metadata-schema namespace-property in the datastore."""
    __tablename__ = 'metadef_namespace_resource_types'
    __table_args__ = (Index('ix_metadef_ns_res_types_namespace_id', 'namespace_id'),)
    resource_type_id = Column(Integer, ForeignKey('metadef_resource_types.id'), primary_key=True, nullable=False)
    namespace_id = Column(Integer, ForeignKey('metadef_namespaces.id'), primary_key=True, nullable=False)
    properties_target = Column(String(80))
    prefix = Column(String(80))