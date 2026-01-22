from __future__ import absolute_import, division, print_function
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, find_obj, find_object_by_name
from ansible.module_utils.basic import AnsibleModule
class VMwareDvSwitchInfoManager(PyVmomi):

    def __init__(self, module):
        super(VMwareDvSwitchInfoManager, self).__init__(module)
        self.folder = self.params['folder']
        self.switch_name = self.params['switch_name']
        folder_obj = None
        if self.folder:
            folder_obj = self.content.searchIndex.FindByInventoryPath(self.folder)
            if not folder_obj:
                self.module.fail_json(msg='Failed to find folder specified by %s' % self.folder)
        if self.switch_name:
            self.switch_objs = [find_object_by_name(self.content, self.switch_name, vim.DistributedVirtualSwitch, folder_obj)]
            if None in self.switch_objs:
                self.switch_objs = None
        else:
            self.switch_objs = find_obj(self.content, [vim.DistributedVirtualSwitch], '', first=False)

    def all_info(self):
        distributed_virtual_switches = []
        if not self.switch_objs:
            self.module.exit_json(changed=False, distributed_virtual_switches=distributed_virtual_switches)
        for switch_obj in self.switch_objs:
            pvlans = []
            if switch_obj.config.pvlanConfig:
                for vlan in switch_obj.config.pvlanConfig:
                    pvlans.append({'primaryVlanId': vlan.primaryVlanId, 'secondaryVlanId': vlan.secondaryVlanId, 'pvlanType': vlan.pvlanType})
            host_members = []
            if switch_obj.summary.hostMember:
                for host in switch_obj.summary.hostMember:
                    host_members.append(host.name)
            health_check = {}
            for health_config in switch_obj.config.healthCheckConfig:
                if isinstance(health_config, vim.dvs.VmwareDistributedVirtualSwitch.VlanMtuHealthCheckConfig):
                    health_check['VlanMtuHealthCheckConfig'] = health_config.enable
                elif isinstance(health_config, vim.dvs.VmwareDistributedVirtualSwitch.TeamingHealthCheckConfig):
                    health_check['TeamingHealthCheckConfig'] = health_config.enable
            distributed_virtual_switches.append({'configure': {'settings': {'properties': {'general': {'name': switch_obj.name, 'vendor': switch_obj.config.productInfo.vendor, 'version': switch_obj.config.productInfo.version, 'numUplinks': len(switch_obj.config.uplinkPortPolicy.uplinkPortName), 'numPorts': switch_obj.summary.numPorts, 'ioControl': switch_obj.config.networkResourceManagementEnabled}, 'advanced': {'maxMtu': switch_obj.config.maxMtu, 'multicastFilteringMode': switch_obj.config.multicastFilteringMode}, 'discoveryProtocol': {'protocol': switch_obj.config.linkDiscoveryProtocolConfig.protocol, 'operation': switch_obj.config.linkDiscoveryProtocolConfig.operation}, 'administratorContact': {'name': switch_obj.config.contact.name, 'contact': switch_obj.config.contact.contact}}, 'privateVlan': pvlans, 'netflow': {'switchIpAddress': switch_obj.config.switchIpAddress, 'collectorIpAddress': switch_obj.config.ipfixConfig.collectorIpAddress, 'collectorPort': switch_obj.config.ipfixConfig.collectorPort, 'observationDomainId': switch_obj.config.ipfixConfig.observationDomainId, 'activeFlowTimeout': switch_obj.config.ipfixConfig.activeFlowTimeout, 'idleFlowTimeout': switch_obj.config.ipfixConfig.idleFlowTimeout, 'samplingRate': switch_obj.config.ipfixConfig.samplingRate, 'internalFlowsOnly': switch_obj.config.ipfixConfig.internalFlowsOnly}, 'healthCheck': health_check}, 'hosts': host_members, 'folder': switch_obj.parent.name, 'name': switch_obj.name}, 'uuid': switch_obj.uuid})
        self.module.exit_json(changed=False, distributed_virtual_switches=distributed_virtual_switches)

    def properties_facts(self):
        distributed_virtual_switches = []
        for switch_obj in self.switch_objs:
            distributed_virtual_switches.append(self.to_json(switch_obj, self.params.get('properties')))
        self.module.exit_json(changed=False, distributed_virtual_switches=distributed_virtual_switches)