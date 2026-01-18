from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
def update_ldap_user_group(module):
    """ Update ldap user group by deleting and creating the LDAP user"""
    changed = delete_ldap_user_group(module)
    if not changed:
        module.fail_json(msg='Cannot delete LDAP user {ldap_group_name}. Cannot find ID for LDAP group.')
    create_ldap_user_group(module)
    changed = True
    return changed