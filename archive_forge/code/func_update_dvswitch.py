from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
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