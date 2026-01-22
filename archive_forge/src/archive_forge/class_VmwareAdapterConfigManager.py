from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
from time import sleep
class VmwareAdapterConfigManager(PyVmomi):
    """Class to configure SR-IOV settings"""

    def __init__(self, module):
        super(VmwareAdapterConfigManager, self).__init__(module)
        cluster_name = self.params.get('cluster_name', None)
        esxi_host_name = self.params.get('esxi_hostname', None)
        self.vmnic = self.params.get('vmnic', None)
        self.num_virt_func = self.params.get('num_virt_func', None)
        self.sriov_on = self.params.get('sriov_on', None)
        self.hosts = self.get_all_host_objs(cluster_name=cluster_name, esxi_host_name=esxi_host_name)
        if not self.hosts:
            self.module.fail_json(msg='Failed to find host system.')
        self.results = {'before': {}, 'after': {}, 'changes': {}}

    def sanitize_params(self):
        """checks user input, raise error if input incompatible
        :return : None
        """
        if self.num_virt_func < 0:
            self.module.fail_json(msg='allowed value for num_virt_func >= 0')
        if self.num_virt_func == 0:
            if self.sriov_on is True:
                self.module.fail_json(msg='with sriov_on == true, allowed value for num_virt_func > 0')
            self.sriov_on = False
        if self.num_virt_func > 0:
            if self.sriov_on is False:
                self.module.fail_json(msg='with sriov_on == false, allowed value for num_virt_func is 0')
            self.sriov_on = True

    def check_compatibility(self, before, hostname):
        """
        checks hardware compatibility with user input, raise error if input incompatible
        :before     : dict, of params on target interface before changing
        :hostname   : str, hosthame
        :return     : None
        """
        if self.num_virt_func > 0:
            if not before['sriovCapable']:
                self.module.fail_json(msg='sriov not supported on host= %s, nic= %s' % (hostname, self.vmnic))
        if before['maxVirtualFunctionSupported'] < self.num_virt_func:
            self.module.fail_json(msg='maxVirtualFunctionSupported= %d on %s' % (before['maxVirtualFunctionSupported'], self.vmnic))

    def make_diff(self, before, hostname):
        """
        preparing diff - changes which will be applied
        :before     : dict, of params on target interface before changing
        :hostname   : str, hosthame
        :return     : dict, of changes which is going to apply
        """
        diff = {}
        change = False
        change_msg = ''
        if before['sriovEnabled'] != self.sriov_on:
            diff['sriovEnabled'] = self.sriov_on
            change = True
        if before['numVirtualFunction'] != self.num_virt_func:
            if before['numVirtualFunctionRequested'] != self.num_virt_func:
                diff['numVirtualFunction'] = self.num_virt_func
                change = True
            else:
                change_msg = 'Not active (looks like not rebooted) '
        if not change:
            change_msg += 'No any changes, already configured '
        diff['msg'] = change_msg
        diff['change'] = change
        return diff

    def set_host_state(self):
        """Checking and applying ESXi host configuration one by one,
        from prepared list of hosts in `self.hosts`.
        For every host applied:
        - user input checking done via calling `sanitize_params` method
        - checks hardware compatibility with user input `check_compatibility`
        - conf changes created via `make_diff`
        - changes applied via calling `_update_sriov` method
        - host state before and after via calling `_check_sriov`
        """
        self.sanitize_params()
        change_list = []
        changed = False
        for host in self.hosts:
            self.results['before'][host.name] = {}
            self.results['after'][host.name] = {}
            self.results['changes'][host.name] = {}
            self.results['before'][host.name] = self._check_sriov(host)
            self.check_compatibility(self.results['before'][host.name], host.name)
            diff = self.make_diff(self.results['before'][host.name], host.name)
            self.results['changes'][host.name] = diff
            if not diff['change']:
                change_list.append(False)
                self.results['after'][host.name] = self._check_sriov(host)
                if self.results['before'][host.name]['rebootRequired'] != self.results['after'][host.name]['rebootRequired']:
                    self.results['changes'][host.name]['rebootRequired'] = self.results['after'][host.name]['rebootRequired']
                continue
            success = self._update_sriov(host, self.sriov_on, self.num_virt_func)
            if success:
                change_list.append(True)
            else:
                change_list.append(False)
            self.results['after'][host.name] = self._check_sriov(host)
            self.results['changes'][host.name].update({'rebootRequired': self.results['after'][host.name]['rebootRequired']})
        if any(change_list):
            changed = True
        self.module.exit_json(changed=changed, diff=self.results)

    def _check_sriov(self, host):
        pnic_info = {}
        pnic_info['rebootRequired'] = host.summary.rebootRequired
        for pci_device in host.configManager.pciPassthruSystem.pciPassthruInfo:
            if pci_device.id == self._getPciId(host):
                try:
                    if pci_device.sriovCapable:
                        pnic_info['sriovCapable'] = True
                        pnic_info['sriovEnabled'] = pci_device.sriovEnabled
                        pnic_info['sriovActive'] = pci_device.sriovActive
                        pnic_info['numVirtualFunction'] = pci_device.numVirtualFunction
                        pnic_info['numVirtualFunctionRequested'] = pci_device.numVirtualFunctionRequested
                        pnic_info['maxVirtualFunctionSupported'] = pci_device.maxVirtualFunctionSupported
                    else:
                        pnic_info['sriovCapable'] = False
                except AttributeError:
                    pnic_info['sriovCapable'] = False
                break
        return pnic_info

    def _getPciId(self, host):
        for pnic in host.config.network.pnic:
            if pnic.device == self.vmnic:
                return pnic.pci
        self.module.fail_json(msg='No nic= %s on host= %s' % (self.vmnic, host.name))

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