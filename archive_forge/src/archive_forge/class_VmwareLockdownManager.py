from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
from ansible.module_utils._text import to_native
class VmwareLockdownManager(PyVmomi):

    def __init__(self, module):
        super(VmwareLockdownManager, self).__init__(module)
        if not self.is_vcenter():
            self.module.fail_json(msg='Lockdown operations are performed from vCenter only. hostname %s is an ESXi server. Please specify hostname as vCenter server.' % self.module.params['hostname'])
        cluster_name = self.params.get('cluster_name', None)
        esxi_host_name = self.params.get('esxi_hostname', None)
        self.hosts = self.get_all_host_objs(cluster_name=cluster_name, esxi_host_name=esxi_host_name)

    def ensure(self):
        """
        Function to manage internal state management
        """
        results = dict(changed=False, host_lockdown_exceptions=dict())
        change_list = []
        desired_state = self.params.get('state')
        exception_users = self.params.get('exception_users')
        for host in self.hosts:
            current_exception_users = host.configManager.hostAccessManager.QueryLockdownExceptions()
            current_exception_users.sort()
            new_exception_users = current_exception_users.copy()
            results['host_lockdown_exceptions'][host.name] = dict(previous_exception_users=current_exception_users)
            changed = False
            if desired_state == 'present':
                for user in exception_users:
                    if user not in current_exception_users:
                        new_exception_users.append(user)
                        changed = True
            elif desired_state == 'absent':
                for user in exception_users:
                    if user in current_exception_users:
                        new_exception_users.remove(user)
                        changed = True
            elif desired_state == 'set':
                if set(current_exception_users) != set(exception_users):
                    new_exception_users = exception_users
                    changed = True
            new_exception_users.sort()
            results['host_lockdown_exceptions'][host.name]['desired_exception_users'] = new_exception_users
            results['host_lockdown_exceptions'][host.name]['current_exception_users'] = new_exception_users
            if changed and (not self.module.check_mode):
                try:
                    host.configManager.hostAccessManager.UpdateLockdownExceptions(new_exception_users)
                except vim.fault.HostConfigFault as host_config_fault:
                    self.module.fail_json(msg='Failed to manage lockdown mode for esxi hostname %s : %s' % (host.name, to_native(host_config_fault.msg)))
                except vim.fault.AdminDisabled as admin_disabled:
                    self.module.fail_json(msg='Failed to manage lockdown mode as administrator permission has been disabled for esxi hostname %s : %s' % (host.name, to_native(admin_disabled.msg)))
                except Exception as generic_exception:
                    self.module.fail_json(msg='Failed to manage lockdown mode due to generic exception for esxi hostname %s : %s' % (host.name, to_native(generic_exception)))
            change_list.append(changed)
        if any(change_list):
            results['changed'] = True
        self.module.exit_json(**results)