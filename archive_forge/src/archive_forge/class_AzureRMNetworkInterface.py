from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import (AzureRMModuleBase,
from ansible.module_utils._text import to_native
class AzureRMNetworkInterface(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), location=dict(type='str'), enable_accelerated_networking=dict(type='bool', default=False), create_with_security_group=dict(type='bool', default=True), security_group=dict(type='raw', aliases=['security_group_name']), state=dict(default='present', choices=['present', 'absent']), private_ip_address=dict(type='str'), private_ip_allocation_method=dict(type='str', choices=['Dynamic', 'Static'], default='Dynamic'), public_ip_address_name=dict(type='str', aliases=['public_ip_address', 'public_ip_name']), public_ip=dict(type='bool', default=True), subnet_name=dict(type='str', aliases=['subnet']), virtual_network=dict(type='raw', aliases=['virtual_network_name']), public_ip_allocation_method=dict(type='str', choices=['Dynamic', 'Static'], default='Dynamic'), ip_configurations=dict(type='list', default=[], elements='dict', options=ip_configuration_spec), os_type=dict(type='str', choices=['Windows', 'Linux'], default='Linux'), open_ports=dict(type='list', elements='str'), enable_ip_forwarding=dict(type='bool', aliases=['ip_forwarding'], default=False), dns_servers=dict(type='list', elements='str'))
        required_if = [('state', 'present', ['subnet_name', 'virtual_network'])]
        self.resource_group = None
        self.name = None
        self.location = None
        self.create_with_security_group = None
        self.enable_accelerated_networking = None
        self.security_group = None
        self.private_ip_address = None
        self.private_ip_allocation_method = None
        self.public_ip_address_name = None
        self.public_ip = None
        self.subnet_name = None
        self.virtual_network = None
        self.public_ip_allocation_method = None
        self.state = None
        self.tags = None
        self.os_type = None
        self.open_ports = None
        self.enable_ip_forwarding = None
        self.ip_configurations = None
        self.dns_servers = None
        self.results = dict(changed=False, state=dict())
        super(AzureRMNetworkInterface, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, required_if=required_if)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            setattr(self, key, kwargs[key])
        results = None
        changed = False
        nic = None
        nsg = None
        resource_group = self.get_resource_group(self.resource_group)
        if not self.location:
            self.location = resource_group.location
        self.location = normalize_location_name(self.location)
        self.virtual_network = self.parse_resource_to_dict(self.virtual_network)
        self.security_group = self.parse_resource_to_dict(self.security_group or self.name)
        if self.ip_configurations:
            for config in self.ip_configurations:
                if config.get('application_security_groups'):
                    asgs = []
                    for asg in config['application_security_groups']:
                        asg_resource_id = asg
                        if isinstance(asg, str) and (not is_valid_resource_id(asg)):
                            asg = self.parse_resource_to_dict(asg)
                        if isinstance(asg, dict):
                            asg_resource_id = format_resource_id(val=asg['name'], subscription_id=self.subscription_id, namespace='Microsoft.Network', types='applicationSecurityGroups', resource_group=asg['resource_group'])
                        asgs.append(asg_resource_id)
                    if len(asgs) > 0:
                        config['application_security_groups'] = asgs
        if self.state == 'present' and (not self.ip_configurations):
            self.deprecate('Setting ip_configuration flatten is deprecated and will be removed. Using ip_configurations list to define the ip configuration', version=(2, 9))
            self.ip_configurations = [dict(private_ip_address=self.private_ip_address, private_ip_allocation_method=self.private_ip_allocation_method, public_ip_address_name=self.public_ip_address_name if self.public_ip else None, public_ip_allocation_method=self.public_ip_allocation_method, name='default', primary=True)]
        try:
            self.log('Fetching network interface {0}'.format(self.name))
            nic = self.network_client.network_interfaces.get(self.resource_group, self.name)
            self.log('Network interface {0} exists'.format(self.name))
            self.check_provisioning_state(nic, self.state)
            results = nic_to_dict(nic)
            self.log(results, pretty_print=True)
            nsg = None
            if self.state == 'present':
                update_tags, results['tags'] = self.update_tags(results['tags'])
                if update_tags:
                    changed = True
                if self.create_with_security_group != bool(results.get('network_security_group')):
                    self.log('CHANGED: add or remove network interface {0} network security group'.format(self.name))
                    changed = True
                if self.enable_accelerated_networking != bool(results.get('enable_accelerated_networking')):
                    self.log('CHANGED: Accelerated Networking set to {0} (previously {1})'.format(self.enable_accelerated_networking, results.get('enable_accelerated_networking')))
                    changed = True
                if self.enable_ip_forwarding != bool(results.get('enable_ip_forwarding')):
                    self.log('CHANGED: IP forwarding set to {0} (previously {1})'.format(self.enable_ip_forwarding, results.get('enable_ip_forwarding')))
                    changed = True
                dns_servers_res = results.get('dns_settings').get('dns_servers')
                _dns_servers_set = sorted(self.dns_servers) if isinstance(self.dns_servers, list) else list()
                _dns_servers_res = sorted(dns_servers_res) if isinstance(self.dns_servers, list) else list()
                if _dns_servers_set != _dns_servers_res:
                    self.log('CHANGED: DNS servers set to {0} (previously {1})'.format(', '.join(_dns_servers_set), ', '.join(_dns_servers_res)))
                    changed = True
                if not changed:
                    nsg = self.get_security_group(self.security_group['resource_group'], self.security_group['name'])
                    if nsg and results.get('network_security_group') and (results['network_security_group'].get('id') != nsg.id):
                        self.log('CHANGED: network interface {0} network security group'.format(self.name))
                        changed = True
                if results['ip_configurations'][0]['subnet']['virtual_network_name'] != self.virtual_network['name']:
                    self.log('CHANGED: network interface {0} virtual network name'.format(self.name))
                    changed = True
                if results['ip_configurations'][0]['subnet']['resource_group'] != self.virtual_network['resource_group']:
                    self.log('CHANGED: network interface {0} virtual network resource group'.format(self.name))
                    changed = True
                if results['ip_configurations'][0]['subnet']['name'] != self.subnet_name:
                    self.log('CHANGED: network interface {0} subnet name'.format(self.name))
                    changed = True
                ip_configuration_result = self.construct_ip_configuration_set(results['ip_configurations'])
                ip_configuration_request = self.construct_ip_configuration_set(self.ip_configurations)
                ip_configuration_result_name = [item['name'] for item in ip_configuration_result]
                for item_request in ip_configuration_request:
                    if item_request['name'] not in ip_configuration_result_name:
                        changed = True
                        break
                    else:
                        for item_result in ip_configuration_result:
                            if len(ip_configuration_request) == 1 and len(ip_configuration_result) == 1:
                                item_request['primary'] = True
                            if item_request['name'] == item_result['name'] and item_request != item_result:
                                changed = True
                                break
            elif self.state == 'absent':
                self.log("CHANGED: network interface {0} exists but requested state is 'absent'".format(self.name))
                changed = True
        except ResourceNotFoundError:
            self.log('Network interface {0} does not exist'.format(self.name))
            if self.state == 'present':
                self.log("CHANGED: network interface {0} does not exist but requested state is 'present'".format(self.name))
                changed = True
        self.results['changed'] = changed
        self.results['state'] = results
        if self.check_mode:
            return self.results
        if changed:
            if self.state == 'present':
                subnet = self.network_models.SubResource(id='/subscriptions/{0}/resourceGroups/{1}/providers/Microsoft.Network/virtualNetworks/{2}/subnets/{3}'.format(self.virtual_network['subscription_id'], self.virtual_network['resource_group'], self.virtual_network['name'], self.subnet_name))
                nic_ip_configurations = [self.network_models.NetworkInterfaceIPConfiguration(private_ip_allocation_method=ip_config.get('private_ip_allocation_method'), private_ip_address=ip_config.get('private_ip_address'), private_ip_address_version=ip_config.get('private_ip_address_version'), name=ip_config.get('name'), subnet=subnet, public_ip_address=self.get_or_create_public_ip_address(ip_config), load_balancer_backend_address_pools=[self.network_models.BackendAddressPool(id=self.backend_addr_pool_id(bap_id)) for bap_id in ip_config.get('load_balancer_backend_address_pools')] if ip_config.get('load_balancer_backend_address_pools') else None, application_gateway_backend_address_pools=[self.network_models.ApplicationGatewayBackendAddressPool(id=self.gateway_backend_addr_pool_id(bap_id)) for bap_id in ip_config.get('application_gateway_backend_address_pools')] if ip_config.get('application_gateway_backend_address_pools') else None, primary=ip_config.get('primary'), application_security_groups=[self.network_models.ApplicationSecurityGroup(id=asg_id) for asg_id in ip_config.get('application_security_groups')] if ip_config.get('application_security_groups') else None) for ip_config in self.ip_configurations]
                nsg = self.create_default_securitygroup(self.security_group['resource_group'], self.location, self.security_group['name'], self.os_type, self.open_ports) if self.create_with_security_group else None
                self.log('Creating or updating network interface {0}'.format(self.name))
                nic = self.network_models.NetworkInterface(id=results['id'] if results else None, location=self.location, tags=self.tags, ip_configurations=nic_ip_configurations, enable_accelerated_networking=self.enable_accelerated_networking, enable_ip_forwarding=self.enable_ip_forwarding, network_security_group=nsg)
                if self.dns_servers:
                    dns_settings = self.network_models.NetworkInterfaceDnsSettings(dns_servers=self.dns_servers)
                    nic.dns_settings = dns_settings
                self.results['state'] = self.create_or_update_nic(nic)
            elif self.state == 'absent':
                self.log('Deleting network interface {0}'.format(self.name))
                self.delete_nic()
                self.results['state']['status'] = 'Deleted'
        return self.results

    def get_or_create_public_ip_address(self, ip_config):
        name = ip_config.get('public_ip_address_name')
        if not (self.public_ip and name):
            return None
        pip = self.get_public_ip_address(name)
        if not pip:
            params = self.network_models.PublicIPAddress(location=self.location, public_ip_allocation_method=ip_config.get('public_ip_allocation_method'))
            try:
                poller = self.network_client.public_ip_addresses.begin_create_or_update(self.resource_group, name, params)
                pip = self.get_poller_result(poller)
            except Exception as exc:
                self.fail('Error creating {0} - {1}'.format(name, str(exc)))
        return pip

    def create_or_update_nic(self, nic):
        try:
            poller = self.network_client.network_interfaces.begin_create_or_update(self.resource_group, self.name, nic)
            new_nic = self.get_poller_result(poller)
            return nic_to_dict(new_nic)
        except Exception as exc:
            self.fail('Error creating or updating network interface {0} - {1}'.format(self.name, str(exc)))

    def delete_nic(self):
        try:
            poller = self.network_client.network_interfaces.begin_delete(self.resource_group, self.name)
            self.get_poller_result(poller)
        except Exception as exc:
            self.fail('Error deleting network interface {0} - {1}'.format(self.name, str(exc)))
        return True

    def get_public_ip_address(self, name):
        self.log('Fetching public ip address {0}'.format(name))
        try:
            return self.network_client.public_ip_addresses.get(self.resource_group, name)
        except ResourceNotFoundError as exc:
            return None

    def get_security_group(self, resource_group, name):
        self.log('Fetching security group {0}'.format(name))
        try:
            return self.network_client.network_security_groups.get(resource_group, name)
        except ResourceNotFoundError as exc:
            return None

    def backend_addr_pool_id(self, val):
        if isinstance(val, dict):
            lb = val.get('load_balancer', None)
            name = val.get('name', None)
            if lb and name:
                return resource_id(subscription=self.subscription_id, resource_group=self.resource_group, namespace='Microsoft.Network', type='loadBalancers', name=lb, child_type_1='backendAddressPools', child_name_1=name)
        return val

    def gateway_backend_addr_pool_id(self, val):
        if isinstance(val, dict):
            appgw = val.get('application_gateway', None)
            name = val.get('name', None)
            if appgw and name:
                return resource_id(subscription=self.subscription_id, resource_group=self.resource_group, namespace='Microsoft.Network', type='applicationGateways', name=appgw, child_type_1='backendAddressPools', child_name_1=name)
        return val

    def construct_ip_configuration_set(self, raw):
        configurations = [dict(private_ip_allocation_method=to_native(item.get('private_ip_allocation_method')), public_ip_address_name=to_native(item.get('public_ip_address').get('name')) if item.get('public_ip_address') else to_native(item.get('public_ip_address_name')), primary=item.get('primary'), load_balancer_backend_address_pools=set([to_native(self.backend_addr_pool_id(id)) for id in item.get('load_balancer_backend_address_pools')]) if item.get('load_balancer_backend_address_pools') else None, application_gateway_backend_address_pools=set([to_native(self.gateway_backend_addr_pool_id(id)) for id in item.get('application_gateway_backend_address_pools')]) if item.get('application_gateway_backend_address_pools') else None, application_security_groups=set([to_native(asg_id) for asg_id in item.get('application_security_groups')]) if item.get('application_security_groups') else None, name=to_native(item.get('name'))) for item in raw]
        return configurations