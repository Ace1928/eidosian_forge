from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils._text import to_native
class VMwareUserInfoManager(PyVmomi):
    """Class to manage local user info"""

    def __init__(self, module):
        super(VMwareUserInfoManager, self).__init__(module)
        if self.is_vcenter():
            self.module.fail_json(msg='Failed to get local account manager settings.', details="It seems that '%s' is a vCenter server instead of an ESXi server" % self.module.params['hostname'])

    def gather_user_info(self):
        """Gather info about local users"""
        results = dict(changed=False, local_user_info=[])
        search_string = ''
        exact_match = False
        find_users = True
        find_groups = False
        user_accounts = self.content.userDirectory.RetrieveUserGroups(None, search_string, None, None, exact_match, find_users, find_groups)
        if user_accounts:
            for user in user_accounts:
                temp_user = dict()
                temp_user['user_name'] = user.principal
                temp_user['description'] = user.fullName
                temp_user['group'] = user.group
                temp_user['user_id'] = user.id
                temp_user['shell_access'] = user.shellAccess
                temp_user['role'] = None
                try:
                    permissions = self.content.authorizationManager.RetrieveEntityPermissions(entity=self.content.rootFolder, inherited=False)
                except vmodl.fault.ManagedObjectNotFound as not_found:
                    self.module.fail_json(msg="The entity doesn't exist: %s" % to_native(not_found))
                for permission in permissions:
                    if permission.principal == user.principal:
                        temp_user['role'] = self.get_role_name(permission.roleId, self.content.authorizationManager.roleList)
                        break
                results['local_user_info'].append(temp_user)
        self.module.exit_json(**results)

    @staticmethod
    def get_role_name(role_id, role_list):
        """Get role name from role ID"""
        role_name = None
        if role_id == -5:
            role_name = 'no-access'
        elif role_id == -2:
            role_name = 'read-only'
        elif role_id == -1:
            role_name = 'admin'
        else:
            for role in role_list:
                if role.roleId == role_id:
                    role_name = role.name
                    break
        return role_name