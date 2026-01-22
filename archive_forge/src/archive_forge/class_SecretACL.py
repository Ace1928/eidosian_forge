import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
class SecretACL(ACL):
    """ACL entity for a secret"""
    columns = ACLFormatter.columns + ('Secret ACL Ref',)
    _acl_type = 'secret'
    _parent_entity_path = '/secrets'

    @property
    def acl_type(self):
        return self._acl_type