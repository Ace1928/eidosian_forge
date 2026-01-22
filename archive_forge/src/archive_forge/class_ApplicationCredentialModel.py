import datetime
import sqlalchemy
from keystone.application_credential.backends import base
from keystone.common import password_hashing
from keystone.common import sql
from keystone import exception
from keystone.i18n import _
class ApplicationCredentialModel(sql.ModelBase, sql.ModelDictMixin):
    __tablename__ = 'application_credential'
    attributes = ['internal_id', 'id', 'name', 'secret_hash', 'description', 'user_id', 'project_id', 'system', 'expires_at', 'unrestricted']
    internal_id = sql.Column(sql.Integer, primary_key=True, nullable=False)
    id = sql.Column(sql.String(64), nullable=False)
    name = sql.Column(sql.String(255), nullable=False)
    secret_hash = sql.Column(sql.String(255), nullable=False)
    description = sql.Column(sql.Text())
    user_id = sql.Column(sql.String(64), nullable=False)
    project_id = sql.Column(sql.String(64), nullable=True)
    system = sql.Column(sql.String(64), nullable=True)
    expires_at = sql.Column(sql.DateTimeInt())
    unrestricted = sql.Column(sql.Boolean)
    __table_args__ = (sql.UniqueConstraint('name', 'user_id', name='duplicate_app_cred_constraint'),)
    roles = sqlalchemy.orm.relationship('ApplicationCredentialRoleModel', backref=sqlalchemy.orm.backref('application_credential'), cascade='all, delete-orphan', cascade_backrefs=False)
    access_rules = sqlalchemy.orm.relationship('ApplicationCredentialAccessRuleModel', backref=sqlalchemy.orm.backref('application_credential'), cascade='all, delete-orphan', cascade_backrefs=False)