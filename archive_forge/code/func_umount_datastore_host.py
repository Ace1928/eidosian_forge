from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, find_datastore_by_name, find_obj, wait_for_task
from ansible_collections.community.vmware.plugins.module_utils.vmware_sms import SMS
from ansible.module_utils._text import to_native
def umount_datastore_host(self):
    ds = find_datastore_by_name(self.content, self.datastore_name)
    if not ds:
        self.module.fail_json(msg='No datastore found with name %s' % self.datastore_name)
    if self.module.check_mode is False:
        error_message_umount = 'Cannot umount datastore %s from host %s' % (self.datastore_name, self.esxi.name)
        try:
            self.esxi.configManager.datastoreSystem.RemoveDatastore(ds)
        except (vim.fault.NotFound, vim.fault.HostConfigFault, vim.fault.ResourceInUse) as fault:
            self.module.fail_json(msg='%s: %s' % (error_message_umount, to_native(fault.msg)))
        except Exception as e:
            self.module.fail_json(msg='%s: %s' % (error_message_umount, to_native(e)))
    self.module.exit_json(changed=True, result='Datastore %s on host %s' % (self.datastore_name, self.esxi.name))