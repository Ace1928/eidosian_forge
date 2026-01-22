from __future__ import absolute_import, division, print_function
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.gitlab import (
def add_identities(self, user, identities, overwrite_identities=False):
    changed = False
    if overwrite_identities:
        changed = self.delete_identities(user, identities)
    for identity in identities:
        if identity not in user.identities:
            setattr(user, 'provider', identity['provider'])
            setattr(user, 'extern_uid', identity['extern_uid'])
            if not self._module.check_mode:
                user.save()
            changed = True
    return changed