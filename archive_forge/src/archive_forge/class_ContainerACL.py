import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
class ContainerACL(ACL):
    """ACL entity for a container"""
    columns = ACLFormatter.columns + ('Container ACL Ref',)
    _acl_type = 'container'
    _parent_entity_path = '/containers'

    @property
    def acl_type(self):
        return self._acl_type