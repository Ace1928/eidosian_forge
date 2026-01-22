from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils.basic import AnsibleModule
class VMwareHostAutoStartManager(PyVmomi):

    def __init__(self, module):
        super(VMwareHostAutoStartManager, self).__init__(module)
        self.esxi_hostname = self.params['esxi_hostname']
        self.name = self.params['name']
        self.uuid = self.params['uuid']
        self.moid = self.params['moid']
        self.system_defaults = self.params['system_defaults']
        self.power_info = self.params['power_info']

    def generate_system_defaults_config(self):
        system_defaults_config = vim.host.AutoStartManager.SystemDefaults()
        system_defaults_config.enabled = self.system_defaults['enabled']
        system_defaults_config.startDelay = self.system_defaults['start_delay']
        system_defaults_config.stopAction = self.system_defaults['stop_action']
        system_defaults_config.stopDelay = self.system_defaults['stop_delay']
        system_defaults_config.waitForHeartbeat = self.system_defaults['wait_for_heartbeat']
        return system_defaults_config

    def generate_power_info_config(self):
        power_info_config = vim.host.AutoStartManager.AutoPowerInfo()
        power_info_config.key = self.vm_obj
        power_info_config.startAction = self.power_info['start_action']
        power_info_config.startDelay = self.power_info['start_delay']
        power_info_config.startOrder = self.power_info['start_order']
        power_info_config.stopAction = self.power_info['stop_action']
        power_info_config.stopDelay = self.power_info['stop_delay']
        power_info_config.waitForHeartbeat = self.power_info['wait_for_heartbeat']
        return power_info_config

    def execute(self):
        result = dict(changed=False, diff={'before': {}, 'after': {}})
        host_obj = self.find_hostsystem_by_name(self.esxi_hostname)
        if not host_obj:
            self.module.fail_json(msg='Cannot find the specified ESXi host: %s' % self.esxi_hostname)
        self.vm_obj = None
        if self.name or self.uuid or self.moid:
            self.vm_obj = self.get_vm()
            if not self.vm_obj:
                self.module.fail_json(msg='Cannot find the specified VM: %s' % (self.name or self.uuid or self.moid))
            elif self.esxi_hostname != self.vm_obj.runtime.host.name:
                self.module.fail_json(msg='%s exists on another host: %s' % (self.name or self.uuid or self.moid, self.vm_obj.runtime.host.name))
        system_defaults_config_difference = False
        existing_system_defaults = self.to_json(host_obj.config.autoStart.defaults)
        system_defaults_for_compare = dict(enabled=existing_system_defaults['enabled'], start_delay=existing_system_defaults['startDelay'], stop_action=existing_system_defaults['stopAction'], stop_delay=existing_system_defaults['stopDelay'], wait_for_heartbeat=existing_system_defaults['waitForHeartbeat'])
        if self.system_defaults:
            if 'guestshutdown' == system_defaults_for_compare['stop_action']:
                system_defaults_for_compare['stop_action'] = 'guestShutdown'
            if 'poweroff' == system_defaults_for_compare['stop_action']:
                system_defaults_for_compare['stop_action'] = 'powerOff'
            if system_defaults_for_compare != self.system_defaults:
                result['diff']['before']['system_defaults'] = OrderedDict(sorted(system_defaults_for_compare.items()))
                result['diff']['after']['system_defaults'] = OrderedDict(sorted(self.system_defaults.items()))
                system_defaults_config_difference = True
        vm_power_info_config_difference = False
        existing_vm_power_info = {}
        if system_defaults_for_compare['enabled'] and self.vm_obj:
            for vm_power_info in host_obj.config.autoStart.powerInfo:
                if vm_power_info.key == self.vm_obj:
                    existing_vm_power_info = self.to_json(vm_power_info)
                    break
            if existing_vm_power_info:
                vm_power_info_for_compare = dict(start_action=existing_vm_power_info['startAction'], start_delay=existing_vm_power_info['startDelay'], start_order=existing_vm_power_info['startOrder'], stop_action=existing_vm_power_info['stopAction'], stop_delay=existing_vm_power_info['stopDelay'], wait_for_heartbeat=existing_vm_power_info['waitForHeartbeat'])
            else:
                vm_power_info_for_compare = dict(start_action='none', start_delay=-1, start_order=-1, stop_action='systemDefault', stop_delay=-1, wait_for_heartbeat='systemDefault')
            if vm_power_info_for_compare != self.power_info:
                result['diff']['before']['power_info'] = OrderedDict(sorted(vm_power_info_for_compare.items()))
                result['diff']['after']['power_info'] = OrderedDict(sorted(self.power_info.items()))
                vm_power_info_config_difference = True
        auto_start_manager_config = vim.host.AutoStartManager.Config()
        auto_start_manager_config.powerInfo = []
        if system_defaults_config_difference or vm_power_info_config_difference:
            if system_defaults_config_difference:
                auto_start_manager_config.defaults = self.generate_system_defaults_config()
                result['system_defaults_config'] = self.system_defaults
            if vm_power_info_config_difference:
                auto_start_manager_config.powerInfo = [self.generate_power_info_config()]
                result['power_info_config'] = self.power_info
            if self.module.check_mode:
                result['changed'] = True
                self.module.exit_json(**result)
            try:
                host_obj.configManager.autoStartManager.ReconfigureAutostart(spec=auto_start_manager_config)
                result['changed'] = True
                self.module.exit_json(**result)
            except Exception as e:
                self.module.fail_json(msg=to_native(e))
            self.module.exit_json(**result)
        else:
            self.module.exit_json(**result)