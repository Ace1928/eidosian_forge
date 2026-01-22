from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import raise_from
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def _get_role_perm(self):
    role_permission = self.role_permission
    args = {'roleid': self._get_role_id()}
    rp = self.query_api('listRolePermissions', **args)
    if rp:
        role_permission = rp['rolepermission']
    return role_permission