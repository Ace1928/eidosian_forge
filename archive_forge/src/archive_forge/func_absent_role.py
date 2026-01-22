from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def absent_role(self):
    role = self.get_role()
    if role:
        self.result['changed'] = True
        args = {'id': role['id']}
        if not self.module.check_mode:
            self.query_api('deleteRole', **args)
    return role