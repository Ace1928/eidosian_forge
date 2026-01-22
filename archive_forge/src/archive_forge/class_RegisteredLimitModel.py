import copy
import sqlalchemy
from sqlalchemy.ext.hybrid import hybrid_property
from oslo_db import exception as db_exception
from keystone.common import driver_hints
from keystone.common import sql
from keystone import exception
from keystone.i18n import _
from keystone.limit.backends import base
class RegisteredLimitModel(sql.ModelBase, sql.ModelDictMixin):
    __tablename__ = 'registered_limit'
    attributes = ['internal_id', 'id', 'service_id', 'region_id', 'resource_name', 'default_limit', 'description']
    internal_id = sql.Column(sql.Integer, primary_key=True, nullable=False)
    id = sql.Column(sql.String(length=64), nullable=False, unique=True)
    service_id = sql.Column(sql.String(255), sql.ForeignKey('service.id'))
    region_id = sql.Column(sql.String(64), sql.ForeignKey('region.id'), nullable=True)
    resource_name = sql.Column(sql.String(255))
    default_limit = sql.Column(sql.Integer, nullable=False)
    description = sql.Column(sql.Text())

    def to_dict(self):
        ref = super(RegisteredLimitModel, self).to_dict()
        ref.pop('internal_id')
        return ref