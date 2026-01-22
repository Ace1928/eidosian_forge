import sqlalchemy
from keystone.common import sql
from keystone.models import revoke_model
from keystone.revoke.backends import base
from oslo_db import api as oslo_db_api
class RevocationEvent(sql.ModelBase, sql.ModelDictMixin):
    __tablename__ = 'revocation_event'
    attributes = revoke_model.REVOKE_KEYS
    id = sql.Column(sql.Integer, primary_key=True, nullable=False)
    domain_id = sql.Column(sql.String(64))
    project_id = sql.Column(sql.String(64))
    user_id = sql.Column(sql.String(64))
    role_id = sql.Column(sql.String(64))
    trust_id = sql.Column(sql.String(64))
    consumer_id = sql.Column(sql.String(64))
    access_token_id = sql.Column(sql.String(64))
    issued_before = sql.Column(sql.DateTime(), nullable=False, index=True)
    expires_at = sql.Column(sql.DateTime())
    revoked_at = sql.Column(sql.DateTime(), nullable=False, index=True)
    audit_id = sql.Column(sql.String(32))
    audit_chain_id = sql.Column(sql.String(32))
    __table_args__ = (sql.Index('ix_revocation_event_project_id_issued_before', 'project_id', 'issued_before'), sql.Index('ix_revocation_event_user_id_issued_before', 'user_id', 'issued_before'), sql.Index('ix_revocation_event_audit_id_issued_before', 'audit_id', 'issued_before'))