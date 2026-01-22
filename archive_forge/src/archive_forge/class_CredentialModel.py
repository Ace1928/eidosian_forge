from oslo_db import api as oslo_db_api
from sqlalchemy.ext.hybrid import hybrid_property
from keystone.common import driver_hints
from keystone.common import sql
from keystone.credential.backends import base
from keystone import exception
class CredentialModel(sql.ModelBase, sql.ModelDictMixinWithExtras):
    __tablename__ = 'credential'
    attributes = ['id', 'user_id', 'project_id', 'encrypted_blob', 'type', 'key_hash']
    id = sql.Column(sql.String(64), primary_key=True)
    user_id = sql.Column(sql.String(64), nullable=False)
    project_id = sql.Column(sql.String(64))
    _encrypted_blob = sql.Column('encrypted_blob', sql.Text(), nullable=False)
    type = sql.Column(sql.String(255), nullable=False)
    key_hash = sql.Column(sql.String(64), nullable=False)
    extra = sql.Column(sql.JsonBlob())

    @hybrid_property
    def encrypted_blob(self):
        return self._encrypted_blob

    @encrypted_blob.setter
    def encrypted_blob(self, encrypted_blob):
        if isinstance(encrypted_blob, bytes):
            encrypted_blob = encrypted_blob.decode('utf-8')
        self._encrypted_blob = encrypted_blob