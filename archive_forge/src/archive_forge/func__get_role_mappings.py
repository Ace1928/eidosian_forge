from __future__ import (absolute_import, division, print_function)
import traceback
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _get_role_mappings(module):
    roleMappings = list()
    for roleMapping in module.params['role_mappings']:
        roleMappings.append(otypes.RegistrationRoleMapping(from_=otypes.Role(name=roleMapping['source_name']) if roleMapping['source_name'] else None, to=otypes.Role(name=roleMapping['dest_name']) if roleMapping['dest_name'] else None))
    return roleMappings