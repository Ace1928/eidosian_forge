from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
class VMwareDvSwitch(PyVmomi):
    """Class to manage a Distributed Virtual Switch"""

    def __init__(self, module):
        super(VMwareDvSwitch, self).__init__(module)
        self.dvs = None
        self.switch_name = self.module.params['switch_name']
        self.switch_version = self.module.params['switch_version']
        if self.switch_version is not None:
            available_dvs_versions = self.available_dvs_versions()
            if self.switch_version not in available_dvs_versions:
                self.module.fail_json(msg="Unsupported version '%s'. Supported versions are: %s." % (self.switch_version, ', '.join(available_dvs_versions)))
        folder = self.params['folder']
        if folder:
            self.folder_obj = self.content.searchIndex.FindByInventoryPath(folder)
            if not self.folder_obj:
                self.module.fail_json(msg='Failed to find the folder specified by %(folder)s' % self.params)
        else:
            datacenter_name = self.params.get('datacenter_name')
            datacenter_obj = self.find_datacenter_by_name(datacenter_name)
            if not datacenter_obj:
                self.module.fail_json(msg="Failed to find datacenter '%s' required for managing distributed vSwitch." % datacenter_name)
            self.folder_obj = datacenter_obj.networkFolder
        self.mtu = self.module.params['mtu']
        if not 1280 <= self.mtu <= 9000:
            self.module.fail_json(msg='MTU value should be between 1280 and 9000 (both inclusive), provided %d.' % self.mtu)
        self.multicast_filtering_mode = self.module.params['multicast_filtering_mode']
        self.uplink_quantity = self.module.params['uplink_quantity']
        self.uplink_prefix = self.module.params['uplink_prefix']
        self.discovery_protocol = self.module.params['discovery_proto']
        self.discovery_operation = self.module.params['discovery_operation']
        self.health_check_vlan = self.params['health_check'].get('vlan_mtu')
        self.health_check_vlan_interval = self.params['health_check'].get('vlan_mtu_interval')
        self.health_check_teaming = self.params['health_check'].get('teaming_failover')
        self.health_check_teaming_interval = self.params['health_check'].get('teaming_failover_interval')
        if self.params['contact']:
            self.contact_name = self.params['contact'].get('name')
            self.contact_details = self.params['contact'].get('description')
        else:
            self.contact_name = None
            self.contact_details = None
        self.description = self.module.params['description']
        self.network_policy = self.module.params['network_policy']
        if self.network_policy is None:
            self.network_policy = {}
        self.netFlow_collector_ip = self.module.params['net_flow'].get('collector_ip') or None
        self.netFlow_collector_port = self.module.params['net_flow'].get('collector_port')
        self.netFlow_observation_domain_id = self.module.params['net_flow'].get('observation_domain_id')
        self.netFlow_active_flow_timeout = self.module.params['net_flow'].get('active_flow_timeout')
        self.netFlow_idle_flow_timeout = self.module.params['net_flow'].get('idle_flow_timeout')
        self.netFlow_sampling_rate = self.module.params['net_flow'].get('sampling_rate')
        self.netFlow_internal_flows_only = self.module.params['net_flow'].get('internal_flows_only')
        self.state = self.module.params['state']

    def available_dvs_versions(self):
        """Get the DVS version supported by the vCenter"""
        dvs_mng = self.content.dvSwitchManager
        available_dvs_specs = dvs_mng.QueryAvailableDvsSpec(recommended=True)
        available_dvs_versions = []
        for available_dvs_spec in available_dvs_specs:
            available_dvs_versions.append(available_dvs_spec.version)
        return available_dvs_versions

    def process_state(self):
        """Process the current state of the DVS"""
        dvs_states = {'absent': {'present': self.destroy_dvswitch, 'absent': self.exit_unchanged}, 'present': {'present': self.update_dvswitch, 'absent': self.create_dvswitch}}
        try:
            dvs_states[self.state][self.check_dvs()]()
        except vmodl.RuntimeFault as runtime_fault:
            self.module.fail_json(msg=to_native(runtime_fault.msg))
        except vmodl.MethodFault as method_fault:
            self.module.fail_json(msg=to_native(method_fault.msg))
        except Exception as e:
            self.module.fail_json(msg=to_native(e))

    def check_dvs(self):
        """Check if DVS is present"""
        self.dvs = find_dvs_by_name(self.content, self.switch_name, folder=self.folder_obj)
        if self.dvs is None:
            return 'absent'
        return 'present'

    def create_dvswitch(self):
        """Create a DVS"""
        changed = True
        results = dict(changed=changed)
        spec = vim.DistributedVirtualSwitch.CreateSpec()
        spec.configSpec = vim.dvs.VmwareDistributedVirtualSwitch.ConfigSpec()
        results['dvswitch'] = self.switch_name
        spec.configSpec.name = self.switch_name
        results['mtu'] = self.mtu
        spec.configSpec.maxMtu = self.mtu
        results['discovery_protocol'] = self.discovery_protocol
        results['discovery_operation'] = self.discovery_operation
        spec.configSpec.linkDiscoveryProtocolConfig = self.create_ldp_spec()
        results['contact'] = self.contact_name
        results['contact_details'] = self.contact_details
        if self.contact_name or self.contact_details:
            spec.configSpec.contact = self.create_contact_spec()
        results['description'] = self.description
        if self.description:
            spec.description = self.description
        results['uplink_quantity'] = self.uplink_quantity
        spec.configSpec.uplinkPortPolicy = vim.DistributedVirtualSwitch.NameArrayUplinkPortPolicy()
        for count in range(1, self.uplink_quantity + 1):
            spec.configSpec.uplinkPortPolicy.uplinkPortName.append('%s%d' % (self.uplink_prefix, count))
        results['uplinks'] = spec.configSpec.uplinkPortPolicy.uplinkPortName
        results['version'] = self.switch_version
        if self.switch_version:
            spec.productInfo = self.create_product_spec(self.switch_version)
        if self.module.check_mode:
            result = 'DVS would be created'
        else:
            network_folder = self.folder_obj
            task = network_folder.CreateDVS_Task(spec)
            try:
                wait_for_task(task)
            except TaskError as invalid_argument:
                self.module.fail_json(msg='Failed to create DVS : %s' % to_native(invalid_argument))
            self.dvs = find_dvs_by_name(self.content, self.switch_name)
            changed_multicast = changed_network_policy = False
            spec = vim.dvs.VmwareDistributedVirtualSwitch.ConfigSpec()
            spec.configVersion = self.dvs.config.configVersion
            results['multicast_filtering_mode'] = self.multicast_filtering_mode
            multicast_filtering_mode = self.get_api_mc_filtering_mode(self.multicast_filtering_mode)
            if self.dvs.config.multicastFilteringMode != multicast_filtering_mode:
                changed_multicast = True
                spec.multicastFilteringMode = multicast_filtering_mode
            spec.multicastFilteringMode = self.get_api_mc_filtering_mode(self.multicast_filtering_mode)
            network_policy = self.network_policy
            if 'promiscuous' in network_policy or 'forged_transmits' in network_policy or 'mac_changes' in network_policy:
                results['network_policy'] = {}
                if 'promiscuous' in network_policy:
                    results['network_policy']['promiscuous'] = network_policy['promiscuous']
                if 'forged_transmits' in network_policy:
                    results['network_policy']['forged_transmits'] = network_policy['forged_transmits']
                if 'mac_changes' in network_policy:
                    results['network_policy']['mac_changes'] = network_policy['mac_changes']
                result = self.check_network_policy_config()
                changed_network_policy = result[1]
                if changed_network_policy:
                    if spec.defaultPortConfig is None:
                        spec.defaultPortConfig = vim.dvs.VmwareDistributedVirtualSwitch.VmwarePortConfigPolicy()
                    spec.defaultPortConfig.macManagementPolicy = result[0]
            if self.netFlow_collector_ip is not None:
                results['net_flow_collector_ip'] = self.netFlow_collector_ip
                results['net_flow_collector_port'] = self.netFlow_collector_port
                results['net_flow_observation_domain_id'] = self.netFlow_observation_domain_id
                results['net_flow_active_flow_timeout'] = self.netFlow_active_flow_timeout
                results['net_flow_idle_flow_timeout'] = self.netFlow_idle_flow_timeout
                results['net_flow_sampling_rate'] = self.netFlow_sampling_rate
                results['net_flow_internal_flows_only'] = self.netFlow_internal_flows_only
            result = self.check_netFlow_config()
            changed_netFlow = result[1]
            if changed_netFlow:
                spec.ipfixConfig = result[0]
            if changed_multicast or changed_network_policy or changed_netFlow:
                self.update_dvs_config(self.dvs, spec)
            results['health_check_vlan'] = self.health_check_vlan
            results['health_check_teaming'] = self.health_check_teaming
            results['uuid'] = self.dvs.uuid
            result = self.check_health_check_config(self.dvs.config.healthCheckConfig)
            changed_health_check = result[1]
            if changed_health_check:
                self.update_health_check_config(self.dvs, result[0])
            result = 'DVS created'
        self.module.exit_json(changed=changed, result=to_native(result))

    def create_ldp_spec(self):
        """Create Link Discovery Protocol config spec"""
        ldp_config_spec = vim.host.LinkDiscoveryProtocolConfig()
        if self.discovery_protocol == 'disabled':
            ldp_config_spec.protocol = 'cdp'
            ldp_config_spec.operation = 'none'
        else:
            ldp_config_spec.protocol = self.discovery_protocol
            ldp_config_spec.operation = self.discovery_operation
        return ldp_config_spec

    def create_product_spec(self, switch_version):
        """Create product info spec"""
        product_info_spec = vim.dvs.ProductSpec()
        product_info_spec.version = switch_version
        return product_info_spec

    @staticmethod
    def get_api_mc_filtering_mode(mode):
        """Get Multicast filtering mode"""
        if mode == 'basic':
            return 'legacyFiltering'
        return 'snooping'

    def create_contact_spec(self):
        """Create contact info spec"""
        contact_info_spec = vim.DistributedVirtualSwitch.ContactInfo()
        contact_info_spec.name = self.contact_name
        contact_info_spec.contact = self.contact_details
        return contact_info_spec

    def update_dvs_config(self, switch_object, spec):
        """Update DVS config"""
        try:
            task = switch_object.ReconfigureDvs_Task(spec)
            wait_for_task(task)
        except TaskError as invalid_argument:
            self.module.fail_json(msg='Failed to update DVS : %s' % to_native(invalid_argument))

    def check_network_policy_config(self):
        changed_promiscuous = changed_forged_transmits = changed_mac_changes = False
        promiscuous_previous = forged_transmits_previous = mac_changes_previous = None
        current_config = self.dvs.config.defaultPortConfig
        policy = vim.dvs.VmwareDistributedVirtualSwitch.MacManagementPolicy()
        if 'promiscuous' in self.network_policy and current_config.macManagementPolicy.allowPromiscuous != self.network_policy['promiscuous']:
            changed_promiscuous = True
            promiscuous_previous = current_config.macManagementPolicy.allowPromiscuous
            policy.allowPromiscuous = self.network_policy['promiscuous']
        if 'forged_transmits' in self.network_policy and current_config.macManagementPolicy.forgedTransmits != self.network_policy['forged_transmits']:
            changed_forged_transmits = True
            forged_transmits_previous = current_config.macManagementPolicy.forgedTransmits
            policy.forgedTransmits = self.network_policy['forged_transmits']
        if 'mac_changes' in self.network_policy and current_config.macManagementPolicy.macChanges != self.network_policy['mac_changes']:
            changed_mac_changes = True
            mac_changes_previous = current_config.macManagementPolicy.macChanges
            policy.macChanges = self.network_policy['mac_changes']
        changed = changed_promiscuous or changed_forged_transmits or changed_mac_changes
        return (policy, changed, changed_promiscuous, promiscuous_previous, changed_forged_transmits, forged_transmits_previous, changed_mac_changes, mac_changes_previous)

    def check_health_check_config(self, health_check_config):
        """Check Health Check config"""
        changed = changed_vlan = changed_vlan_interval = changed_teaming = changed_teaming_interval = False
        vlan_previous = teaming_previous = None
        vlan_interval_previous = teaming_interval_previous = 0
        for config in health_check_config:
            if isinstance(config, vim.dvs.VmwareDistributedVirtualSwitch.VlanMtuHealthCheckConfig):
                if config.enable != self.health_check_vlan:
                    changed = changed_vlan = True
                    vlan_previous = config.enable
                    config.enable = self.health_check_vlan
                if config.enable and config.interval != self.health_check_vlan_interval:
                    changed = changed_vlan_interval = True
                    vlan_interval_previous = config.interval
                    config.interval = self.health_check_vlan_interval
            if isinstance(config, vim.dvs.VmwareDistributedVirtualSwitch.TeamingHealthCheckConfig):
                if config.enable != self.health_check_teaming:
                    changed = changed_teaming = True
                    teaming_previous = config.enable
                    config.enable = self.health_check_teaming
                if config.enable and config.interval != self.health_check_teaming_interval:
                    changed = changed_teaming_interval = True
                    teaming_interval_previous = config.interval
                    config.interval = self.health_check_teaming_interval
        return (health_check_config, changed, changed_vlan, vlan_previous, changed_vlan_interval, vlan_interval_previous, changed_teaming, teaming_previous, changed_teaming_interval, teaming_interval_previous)

    def update_health_check_config(self, switch_object, health_check_config):
        """Update Health Check config"""
        try:
            task = switch_object.UpdateDVSHealthCheckConfig_Task(healthCheckConfig=health_check_config)
        except vim.fault.DvsFault as dvs_fault:
            self.module.fail_json(msg='Update failed due to DVS fault : %s' % to_native(dvs_fault))
        except vmodl.fault.NotSupported as not_supported:
            self.module.fail_json(msg='Health check not supported on the switch : %s' % to_native(not_supported))
        except TaskError as invalid_argument:
            self.module.fail_json(msg='Failed to configure health check : %s' % to_native(invalid_argument))
        try:
            wait_for_task(task)
        except TaskError as invalid_argument:
            self.module.fail_json(msg='Failed to update health check config : %s' % to_native(invalid_argument))

    def check_netFlow_config(self):
        """Check NetFlow config"""
        changed = changed_collectorIpAddress = changed_collectorPort = changed_observationDomainId = changed_activeFlowTimeout = changed_idleFlowTimeout = changed_samplingRate = changed_internalFlowsOnly = False
        collectorIpAddress_previous = collectorPort_previous = observationDomainId_previous = activeFlowTimeout_previous = idleFlowTimeout_previous = samplingRate_previous = internalFlowsOnly_previous = None
        current_config = self.dvs.config.ipfixConfig
        if current_config is None:
            new_config = vim.dvs.VmwareDistributedVirtualSwitch.IpfixConfig()
        else:
            new_config = current_config
        if self.netFlow_collector_ip is not None:
            if current_config.collectorIpAddress != self.netFlow_collector_ip:
                changed = changed_collectorIpAddress = True
                collectorIpAddress_previous = current_config.collectorIpAddress
                new_config.collectorIpAddress = self.netFlow_collector_ip
            if current_config.collectorPort != self.netFlow_collector_port:
                changed = changed_collectorPort = True
                collectorPort_previous = current_config.collectorPort
                new_config.collectorPort = self.netFlow_collector_port
            if current_config.observationDomainId != self.netFlow_observation_domain_id:
                changed = changed_observationDomainId = True
                observationDomainId_previous = current_config.observationDomainId
                new_config.observationDomainId = self.netFlow_observation_domain_id
            if current_config.activeFlowTimeout != self.netFlow_active_flow_timeout:
                changed = changed_activeFlowTimeout = True
                activeFlowTimeout_previous = current_config.activeFlowTimeout
                new_config.activeFlowTimeout = self.netFlow_active_flow_timeout
            if current_config.idleFlowTimeout != self.netFlow_idle_flow_timeout:
                changed = changed_idleFlowTimeout = True
                idleFlowTimeout_previous = current_config.idleFlowTimeout
                new_config.idleFlowTimeout = self.netFlow_idle_flow_timeout
            if current_config.samplingRate != self.netFlow_sampling_rate:
                changed = changed_samplingRate = True
                samplingRate_previous = current_config.samplingRate
                new_config.samplingRate = self.netFlow_sampling_rate
            if current_config.internalFlowsOnly != self.netFlow_internal_flows_only:
                changed = changed_internalFlowsOnly = True
                internalFlowsOnly_previous = current_config.internalFlowsOnly
                new_config.internalFlowsOnly = self.netFlow_internal_flows_only
        return (new_config, changed, changed_collectorIpAddress, collectorIpAddress_previous, changed_collectorPort, collectorPort_previous, changed_observationDomainId, observationDomainId_previous, changed_activeFlowTimeout, activeFlowTimeout_previous, changed_idleFlowTimeout, idleFlowTimeout_previous, changed_samplingRate, samplingRate_previous, changed_internalFlowsOnly, internalFlowsOnly_previous)

    def exit_unchanged(self):
        """Exit with status message"""
        changed = False
        results = dict(changed=changed)
        results['dvswitch'] = self.switch_name
        results['result'] = 'DVS not present'
        self.module.exit_json(**results)

    def destroy_dvswitch(self):
        """Delete a DVS"""
        changed = True
        results = dict(changed=changed)
        results['dvswitch'] = self.switch_name
        if self.module.check_mode:
            results['result'] = 'DVS would be deleted'
        else:
            try:
                task = self.dvs.Destroy_Task()
            except vim.fault.VimFault as vim_fault:
                self.module.fail_json(msg='Failed to deleted DVS : %s' % to_native(vim_fault))
            wait_for_task(task)
            results['result'] = 'DVS deleted'
        self.module.exit_json(**results)

    def update_dvswitch(self):
        """Check and update DVS settings"""
        changed = changed_settings = changed_ldp = changed_version = changed_health_check = changed_network_policy = changed_netFlow = False
        results = dict(changed=changed)
        results['dvswitch'] = self.switch_name
        changed_list = []
        message = ''
        config_spec = vim.dvs.VmwareDistributedVirtualSwitch.ConfigSpec()
        config_spec.configVersion = self.dvs.config.configVersion
        results['mtu'] = self.mtu
        if self.dvs.config.maxMtu != self.mtu:
            changed = changed_settings = True
            changed_list.append('mtu')
            results['mtu_previous'] = config_spec.maxMtu
            config_spec.maxMtu = self.mtu
        ldp_protocol = self.dvs.config.linkDiscoveryProtocolConfig.protocol
        ldp_operation = self.dvs.config.linkDiscoveryProtocolConfig.operation
        if self.discovery_protocol == 'disabled':
            results['discovery_protocol'] = self.discovery_protocol
            results['discovery_operation'] = 'n/a'
            if ldp_protocol != 'cdp' or ldp_operation != 'none':
                changed_ldp = True
                results['discovery_protocol_previous'] = ldp_protocol
                results['discovery_operation_previous'] = ldp_operation
        else:
            results['discovery_protocol'] = self.discovery_protocol
            results['discovery_operation'] = self.discovery_operation
            if ldp_protocol != self.discovery_protocol or ldp_operation != self.discovery_operation:
                changed_ldp = True
                if ldp_protocol != self.discovery_protocol:
                    results['discovery_protocol_previous'] = ldp_protocol
                if ldp_operation != self.discovery_operation:
                    results['discovery_operation_previous'] = ldp_operation
        if changed_ldp:
            changed = changed_settings = True
            changed_list.append('discovery protocol')
            config_spec.linkDiscoveryProtocolConfig = self.create_ldp_spec()
        results['multicast_filtering_mode'] = self.multicast_filtering_mode
        multicast_filtering_mode = self.get_api_mc_filtering_mode(self.multicast_filtering_mode)
        if self.dvs.config.multicastFilteringMode != multicast_filtering_mode:
            changed = changed_settings = True
            changed_list.append('multicast filtering')
            results['multicast_filtering_mode_previous'] = self.dvs.config.multicastFilteringMode
            config_spec.multicastFilteringMode = multicast_filtering_mode
        results['contact'] = self.contact_name
        results['contact_details'] = self.contact_details
        if self.dvs.config.contact.name != self.contact_name or self.dvs.config.contact.contact != self.contact_details:
            changed = changed_settings = True
            changed_list.append('contact')
            results['contact_previous'] = self.dvs.config.contact.name
            results['contact_details_previous'] = self.dvs.config.contact.contact
            config_spec.contact = self.create_contact_spec()
        results['description'] = self.description
        if self.dvs.config.description != self.description:
            changed = changed_settings = True
            changed_list.append('description')
            results['description_previous'] = self.dvs.config.description
            if self.description is None:
                config_spec.description = ''
            else:
                config_spec.description = self.description
        results['uplink_quantity'] = self.uplink_quantity
        if len(self.dvs.config.uplinkPortPolicy.uplinkPortName) != self.uplink_quantity:
            changed = changed_settings = True
            changed_list.append('uplink quantity')
            results['uplink_quantity_previous'] = len(self.dvs.config.uplinkPortPolicy.uplinkPortName)
            config_spec.uplinkPortPolicy = vim.DistributedVirtualSwitch.NameArrayUplinkPortPolicy()
            if len(self.dvs.config.uplinkPortPolicy.uplinkPortName) < self.uplink_quantity:
                for count in range(1, self.uplink_quantity + 1):
                    config_spec.uplinkPortPolicy.uplinkPortName.append('%s%d' % (self.uplink_prefix, count))
            if len(self.dvs.config.uplinkPortPolicy.uplinkPortName) > self.uplink_quantity:
                for count in range(1, self.uplink_quantity + 1):
                    config_spec.uplinkPortPolicy.uplinkPortName.append('%s%d' % (self.uplink_prefix, count))
            results['uplinks'] = config_spec.uplinkPortPolicy.uplinkPortName
            results['uplinks_previous'] = self.dvs.config.uplinkPortPolicy.uplinkPortName
        else:
            results['uplinks'] = self.dvs.config.uplinkPortPolicy.uplinkPortName
        results['health_check_vlan'] = self.health_check_vlan
        results['health_check_teaming'] = self.health_check_teaming
        results['health_check_vlan_interval'] = self.health_check_vlan_interval
        results['health_check_teaming_interval'] = self.health_check_teaming_interval
        health_check_config, changed_health_check, changed_vlan, vlan_previous, changed_vlan_interval, vlan_interval_previous, changed_teaming, teaming_previous, changed_teaming_interval, teaming_interval_previous = self.check_health_check_config(self.dvs.config.healthCheckConfig)
        if changed_health_check:
            changed = True
            changed_list.append('health check')
            if changed_vlan:
                results['health_check_vlan_previous'] = vlan_previous
            if changed_vlan_interval:
                results['health_check_vlan_interval_previous'] = vlan_interval_previous
            if changed_teaming:
                results['health_check_teaming_previous'] = teaming_previous
            if changed_teaming_interval:
                results['health_check_teaming_interval_previous'] = teaming_interval_previous
        if 'promiscuous' in self.network_policy or 'forged_transmits' in self.network_policy or 'mac_changes' in self.network_policy:
            results['network_policy'] = {}
            if 'promiscuous' in self.network_policy:
                results['network_policy']['promiscuous'] = self.network_policy['promiscuous']
            if 'forged_transmits' in self.network_policy:
                results['network_policy']['forged_transmits'] = self.network_policy['forged_transmits']
            if 'mac_changes' in self.network_policy:
                results['network_policy']['mac_changes'] = self.network_policy['mac_changes']
            policy, changed_network_policy, changed_promiscuous, promiscuous_previous, changed_forged_transmits, forged_transmits_previous, changed_mac_changes, mac_changes_previous = self.check_network_policy_config()
            if changed_network_policy:
                changed = changed_settings = True
                changed_list.append('network policy')
                results['network_policy_previous'] = {}
                if changed_promiscuous:
                    results['network_policy_previous']['promiscuous'] = promiscuous_previous
                if changed_forged_transmits:
                    results['network_policy_previous']['forged_transmits'] = forged_transmits_previous
                if changed_mac_changes:
                    results['network_policy_previous']['mac_changes'] = mac_changes_previous
                if config_spec.defaultPortConfig is None:
                    config_spec.defaultPortConfig = vim.dvs.VmwareDistributedVirtualSwitch.VmwarePortConfigPolicy()
                config_spec.defaultPortConfig.macManagementPolicy = policy
        if self.switch_version:
            results['version'] = self.switch_version
            if self.dvs.config.productInfo.version != self.switch_version:
                changed_version = True
                spec_product = self.create_product_spec(self.switch_version)
        else:
            results['version'] = self.dvs.config.productInfo.version
            changed_version = False
        if changed_version:
            changed = True
            changed_list.append('switch version')
            results['version_previous'] = self.dvs.config.productInfo.version
        if self.netFlow_collector_ip is not None:
            results['net_flow_collector_ip'] = self.netFlow_collector_ip
            results['net_flow_collector_port'] = self.netFlow_collector_port
            results['net_flow_observation_domain_id'] = self.netFlow_observation_domain_id
            results['net_flow_active_flow_timeout'] = self.netFlow_active_flow_timeout
            results['net_flow_idle_flow_timeout'] = self.netFlow_idle_flow_timeout
            results['net_flow_sampling_rate'] = self.netFlow_sampling_rate
            results['net_flow_internal_flows_only'] = self.netFlow_internal_flows_only
        ipfixConfig, changed_netFlow, changed_collectorIpAddress, collectorIpAddress_previous, changed_collectorPort, collectorPort_previous, changed_observationDomainId, observationDomainId_previous, changed_activeFlowTimeout, activeFlowTimeout_previous, changed_idleFlowTimeout, idleFlowTimeout_previous, changed_samplingRate, samplingRate_previous, changed_internalFlowsOnly, internalFlowsOnly_previous = self.check_netFlow_config()
        if changed_netFlow:
            changed = changed_settings = True
            changed_list.append('netFlow')
            if changed_collectorIpAddress:
                results['net_flow_collector_ip_previous'] = collectorIpAddress_previous
            if changed_collectorPort:
                results['net_flow_collector_port_previous'] = collectorPort_previous
            if changed_observationDomainId:
                results['net_flow_observation_domain_id_previous'] = observationDomainId_previous
            if changed_activeFlowTimeout:
                results['net_flow_active_flow_timeout_previous'] = activeFlowTimeout_previous
            if changed_idleFlowTimeout:
                results['net_flow_idle_flow_timeout_previous'] = idleFlowTimeout_previous
            if changed_samplingRate:
                results['net_flow_sampling_rate_previous'] = samplingRate_previous
            if changed_internalFlowsOnly:
                results['net_flow_internal_flows_only_previous'] = internalFlowsOnly_previous
            config_spec.ipfixConfig = ipfixConfig
        if changed:
            if self.module.check_mode:
                changed_suffix = ' would be changed'
            else:
                changed_suffix = ' changed'
            if len(changed_list) > 2:
                message = ', '.join(changed_list[:-1]) + ', and ' + str(changed_list[-1])
            elif len(changed_list) == 2:
                message = ' and '.join(changed_list)
            elif len(changed_list) == 1:
                message = changed_list[0]
            message += changed_suffix
            if not self.module.check_mode:
                if changed_settings:
                    self.update_dvs_config(self.dvs, config_spec)
                if changed_health_check:
                    self.update_health_check_config(self.dvs, health_check_config)
                if changed_version:
                    task = self.dvs.PerformDvsProductSpecOperation_Task('upgrade', spec_product)
                    try:
                        wait_for_task(task)
                    except TaskError as invalid_argument:
                        self.module.fail_json(msg='Failed to update DVS version : %s' % to_native(invalid_argument))
        else:
            message = 'DVS already configured properly'
        results['uuid'] = self.dvs.uuid
        results['changed'] = changed
        results['result'] = message
        self.module.exit_json(**results)