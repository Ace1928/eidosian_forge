from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
class AzureRMPrivateEndpointDnsZoneGroup(AzureRMModuleBaseExt):

    def __init__(self):
        self.module_arg_spec = dict(name=dict(type='str', required=True), private_endpoint=dict(type='str', required=True), resource_group=dict(type='str', required=True), private_dns_zone_configs=dict(type='list', elements='dict', options=private_dns_zone_configs_spec), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.name = None
        self.private_endpoint = None
        self.resource_group = None
        self.state = None
        self.parameters = dict()
        self.results = dict(changed=False, state=dict())
        self.to_do = Actions.NoAction
        super(AzureRMPrivateEndpointDnsZoneGroup, self).__init__(self.module_arg_spec, supports_tags=False, supports_check_mode=True)

    def exec_module(self, **kwargs):
        for key in self.module_arg_spec:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            elif kwargs[key] is not None:
                self.parameters[key] = kwargs[key]
        for zone_config in self.parameters.get('private_dns_zone_configs', []):
            if zone_config.get('private_dns_zone_id') is not None:
                self.log('The private_dns_zone_id exist, do nothing')
            else:
                zone_name = zone_config.pop('private_dns_zone')
                zone_config['private_dns_zone_id'] = self.private_dns_zone_id(zone_name)
        self.log('Fetching private endpoint {0}'.format(self.name))
        old_response = self.get_zone()
        if old_response is None or not old_response:
            if self.state == 'present':
                self.to_do = Actions.Create
                self.ensure_private_endpoint()
        elif self.state == 'absent':
            self.to_do = Actions.Delete
        else:
            self.results['compare'] = []
            if not self.idempotency_check(old_response, self.parameters):
                self.to_do = Actions.Update
        if self.to_do == Actions.Create or self.to_do == Actions.Update:
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            response = self.create_update_zone()
        elif self.to_do == Actions.Delete:
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            response = self.delete_zone()
        else:
            self.results['changed'] = False
            response = old_response
        if response is not None:
            self.results['state'] = response
        return self.results

    def get_zone(self):
        try:
            item = self.network_client.private_dns_zone_groups.get(resource_group_name=self.resource_group, private_endpoint_name=self.private_endpoint, private_dns_zone_group_name=self.name)
            return self.zone_to_dict(item)
        except ResourceNotFoundError:
            self.log('Did not find the private endpoint resource')
        return None

    def create_update_zone(self):
        try:
            self.parameters['name'] = self.name
            response = self.network_client.private_dns_zone_groups.begin_create_or_update(resource_group_name=self.resource_group, private_endpoint_name=self.private_endpoint, private_dns_zone_group_name=self.name, parameters=self.parameters)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
            return self.zone_to_dict(response)
        except Exception as exc:
            self.fail('Error creating or updating DNS zone group {0} for private endpoint {1}: {2}'.format(self.name, self.private_endpoint, str(exc)))

    def ensure_private_endpoint(self):
        try:
            self.network_client.private_endpoints.get(resource_group_name=self.resource_group, private_endpoint_name=self.private_endpoint)
        except ResourceNotFoundError:
            self.fail('Could not load the private endpoint {0}.'.format(self.private_endpoint))

    def delete_zone(self):
        try:
            response = self.network_client.private_dns_zone_groups.begin_delete(resource_group_name=self.resource_group, private_endpoint_name=self.private_endpoint, private_dns_zone_group_name=self.name)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
            return response
        except Exception as exc:
            self.fail('Error deleting private endpoint {0}: {1}'.format(self.name, str(exc)))

    def zone_to_dict(self, zone):
        if zone is not None:
            zone_dict = zone.as_dict()
        else:
            return None
        return dict(id=zone_dict.get('id'), name=zone_dict.get('name'), private_dns_zone_configs=[self.zone_config_to_dict(zone_config) for zone_config in zone_dict.get('private_dns_zone_configs', [])], provisioning_state=zone_dict.get('provisioning_state'))

    def zone_config_to_dict(self, zone_config):
        return dict(name=zone_config.get('name'), private_dns_zone_id=zone_config.get('private_dns_zone_id'), record_sets=[self.record_set_to_dict(record_set) for record_set in zone_config.get('record_sets', [])])

    def record_set_to_dict(self, record_set):
        return dict(fqdn=record_set.get('fqdn'), ip_addresses=record_set.get('ip_addresses'), provisioning_state=record_set.get('provisioning_state'), record_set_name=record_set.get('record_set_name'), record_type=record_set.get('record_type'), ttl=record_set.get('ttl'))

    def private_dns_zone_id(self, name):
        return resource_id(subscription=self.subscription_id, resource_group=self.resource_group, namespace='Microsoft.Network', type='privateDnsZones', name=name)