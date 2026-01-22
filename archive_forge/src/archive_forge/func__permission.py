from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _permission(module, permissions_service, connection):
    for permission in permissions_service.list():
        user = follow_link(connection, permission.user)
        if equal(module.params['user_name'], user.principal if user else None) and equal(module.params['group_name'], get_link_name(connection, permission.group)) and equal(module.params['role'], get_link_name(connection, permission.role)):
            return permission