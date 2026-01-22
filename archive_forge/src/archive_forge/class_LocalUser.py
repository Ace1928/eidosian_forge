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
class LocalUser(sql.ModelBase, sql.ModelDictMixin):
    __tablename__ = 'local_user'
    attributes = ['id', 'user_id', 'domain_id', 'name']
    id = sql.Column(sql.Integer, primary_key=True)
    user_id = sql.Column(sql.String(64), nullable=False)
    domain_id = sql.Column(sql.String(64), nullable=False)
    name = sql.Column(sql.String(255), nullable=False)
    passwords = orm.relationship('Password', single_parent=True, cascade='all,delete-orphan', lazy='joined', backref='local_user', order_by='Password.created_at_int')
    failed_auth_count = sql.Column(sql.Integer, nullable=True)
    failed_auth_at = sql.Column(sql.DateTime, nullable=True)
    __table_args__ = (sql.UniqueConstraint('user_id'), sql.UniqueConstraint('domain_id', 'name'), sqlalchemy.ForeignKeyConstraint(['user_id', 'domain_id'], ['user.id', 'user.domain_id'], onupdate='CASCADE', ondelete='CASCADE'))