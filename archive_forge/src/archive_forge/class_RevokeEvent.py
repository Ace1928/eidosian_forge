from oslo_log import log
from oslo_serialization import msgpackutils
from oslo_utils import timeutils
from keystone.common import cache
from keystone.common import utils
class RevokeEvent(object):

    def __init__(self, **kwargs):
        for k in REVOKE_KEYS:
            v = kwargs.get(k)
            setattr(self, k, v)
        if self.domain_id and self.expires_at:
            self.domain_scope_id = self.domain_id
            self.domain_id = None
        else:
            self.domain_scope_id = None
        if self.expires_at is not None:
            self.expires_at = self.expires_at.replace(microsecond=0)
        if self.revoked_at is None:
            self.revoked_at = timeutils.utcnow().replace(microsecond=0)
        if self.issued_before is None:
            self.issued_before = self.revoked_at

    def to_dict(self):
        keys = ['user_id', 'role_id', 'domain_id', 'domain_scope_id', 'project_id', 'audit_id', 'audit_chain_id']
        event = {key: self.__dict__[key] for key in keys if self.__dict__[key] is not None}
        if self.trust_id is not None:
            event['OS-TRUST:trust_id'] = self.trust_id
        if self.consumer_id is not None:
            event['OS-OAUTH1:consumer_id'] = self.consumer_id
        if self.access_token_id is not None:
            event['OS-OAUTH1:access_token_id'] = self.access_token_id
        if self.expires_at is not None:
            event['expires_at'] = utils.isotime(self.expires_at)
        if self.issued_before is not None:
            event['issued_before'] = utils.isotime(self.issued_before, subsecond=True)
        if self.revoked_at is not None:
            event['revoked_at'] = utils.isotime(self.revoked_at, subsecond=True)
        return event