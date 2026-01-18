from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import raise_from
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def update_role_perm(self, role_perm):
    perm_order = None
    if not self.module.params.get('parent'):
        args = {'ruleid': role_perm['id'], 'roleid': role_perm['roleid'], 'permission': self.module.params.get('permission')}
        if self.has_changed(args, role_perm, only_keys=['permission']):
            self.result['changed'] = True
            if not self.module.check_mode:
                if self.cloudstack_version >= LooseVersion('4.11.0'):
                    self.query_api('updateRolePermission', **args)
                    role_perm = self._get_rule()
                else:
                    perm_order = self.replace_rule()
    else:
        perm_order = self.order_permissions(self.module.params.get('parent'), role_perm['id'])
    if perm_order:
        args = {'roleid': role_perm['roleid'], 'ruleorder': perm_order}
        self.result['changed'] = True
        if not self.module.check_mode:
            self.query_api('updateRolePermission', **args)
            role_perm = self._get_rule()
    return role_perm