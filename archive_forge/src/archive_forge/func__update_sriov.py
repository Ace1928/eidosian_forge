from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
from time import sleep
def _update_sriov(self, host, sriovEnabled, numVirtualFunction):
    nic_sriov = vim.host.SriovConfig()
    nic_sriov.id = self._getPciId(host)
    nic_sriov.sriovEnabled = sriovEnabled
    nic_sriov.numVirtualFunction = numVirtualFunction
    try:
        if not self.module.check_mode:
            host.configManager.pciPassthruSystem.UpdatePassthruConfig([nic_sriov])
            host.configManager.pciPassthruSystem.Refresh()
            sleep(2)
            return True
        return False
    except vim.fault.HostConfigFault as config_fault:
        self.module.fail_json(msg='Failed to configure SR-IOV for host= %s due to : %s' % (host.name, to_native(config_fault.msg)))
        return False