import datetime
import sqlalchemy
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy import orm
from sqlalchemy.orm import collections
from keystone.common import password_hashing
from keystone.common import resource_options
from keystone.common import sql
import keystone.conf
from keystone.identity.backends import resource_options as iro
class FederatedUser(sql.ModelBase, sql.ModelDictMixin):
    __tablename__ = 'federated_user'
    attributes = ['id', 'user_id', 'idp_id', 'protocol_id', 'unique_id', 'display_name']
    id = sql.Column(sql.Integer, primary_key=True)
    user_id = sql.Column(sql.String(64), sql.ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    idp_id = sql.Column(sql.String(64), sql.ForeignKey('identity_provider.id', ondelete='CASCADE'), nullable=False)
    protocol_id = sql.Column(sql.String(64), nullable=False)
    unique_id = sql.Column(sql.String(255), nullable=False)
    display_name = sql.Column(sql.String(255), nullable=True)
    __table_args__ = (sql.UniqueConstraint('idp_id', 'protocol_id', 'unique_id'), sqlalchemy.ForeignKeyConstraint(['protocol_id', 'idp_id'], ['federation_protocol.id', 'federation_protocol.idp_id'], ondelete='CASCADE'))