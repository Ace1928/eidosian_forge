from oslo_log import log
from oslo_serialization import jsonutils
from sqlalchemy import orm
from keystone.common import sql
import keystone.conf
from keystone import exception
from keystone.federation.backends import base
from keystone.i18n import _
class IdentityProviderModel(sql.ModelBase, sql.ModelDictMixin):
    __tablename__ = 'identity_provider'
    attributes = ['id', 'domain_id', 'enabled', 'description', 'remote_ids', 'authorization_ttl']
    mutable_attributes = frozenset(['description', 'enabled', 'remote_ids', 'authorization_ttl'])
    id = sql.Column(sql.String(64), primary_key=True)
    domain_id = sql.Column(sql.String(64), nullable=False)
    enabled = sql.Column(sql.Boolean, nullable=False)
    description = sql.Column(sql.Text(), nullable=True)
    authorization_ttl = sql.Column(sql.Integer, nullable=True)
    remote_ids = orm.relationship('IdPRemoteIdsModel', order_by='IdPRemoteIdsModel.remote_id', cascade='all, delete-orphan')
    expiring_user_group_memberships = orm.relationship('ExpiringUserGroupMembership', cascade='all, delete-orphan', backref='idp')

    @classmethod
    def from_dict(cls, dictionary):
        new_dictionary = dictionary.copy()
        remote_ids_list = new_dictionary.pop('remote_ids', None)
        if not remote_ids_list:
            remote_ids_list = []
        identity_provider = cls(**new_dictionary)
        remote_ids = []
        for remote in remote_ids_list:
            remote_ids.append(IdPRemoteIdsModel(remote_id=remote))
        identity_provider.remote_ids = remote_ids
        return identity_provider

    def to_dict(self):
        """Return a dictionary with model's attributes."""
        d = dict()
        for attr in self.__class__.attributes:
            d[attr] = getattr(self, attr)
        d['remote_ids'] = []
        for remote in self.remote_ids:
            d['remote_ids'].append(remote.remote_id)
        return d