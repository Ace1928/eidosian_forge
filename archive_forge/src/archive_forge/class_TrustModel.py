from oslo_utils import timeutils
import sqlalchemy
from sqlalchemy.ext.hybrid import hybrid_property
from keystone.common import sql
from keystone import exception
from keystone.trust.backends import base
class TrustModel(sql.ModelBase, sql.ModelDictMixinWithExtras):
    __tablename__ = 'trust'
    attributes = ['id', 'trustor_user_id', 'trustee_user_id', 'project_id', 'impersonation', 'expires_at', 'remaining_uses', 'deleted_at', 'redelegated_trust_id', 'redelegation_count']
    id = sql.Column(sql.String(64), primary_key=True)
    trustor_user_id = sql.Column(sql.String(64), nullable=False)
    trustee_user_id = sql.Column(sql.String(64), nullable=False)
    project_id = sql.Column(sql.String(64))
    impersonation = sql.Column(sql.Boolean, nullable=False)
    deleted_at = sql.Column(sql.DateTime)
    _expires_at = sql.Column('expires_at', sql.DateTime)
    expires_at_int = sql.Column(sql.DateTimeInt(), nullable=True)
    remaining_uses = sql.Column(sql.Integer, nullable=True)
    redelegated_trust_id = sql.Column(sql.String(64), nullable=True)
    redelegation_count = sql.Column(sql.Integer, nullable=True)
    extra = sql.Column(sql.JsonBlob())
    __table_args__ = (sql.UniqueConstraint('trustor_user_id', 'trustee_user_id', 'project_id', 'impersonation', 'expires_at', name='duplicate_trust_constraint'),)

    @hybrid_property
    def expires_at(self):
        return self.expires_at_int or self._expires_at

    @expires_at.setter
    def expires_at(self, value):
        self._expires_at = value
        self.expires_at_int = value