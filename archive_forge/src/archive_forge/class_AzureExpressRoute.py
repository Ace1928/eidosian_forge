from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureExpressRoute(AzureRMModuleBase):

    def __init__(self):
        self.service_provider_properties_spec = dict(service_provider_name=dict(type='str'), peering_location=dict(type='str'), bandwidth_in_mbps=dict(type='str'))
        self.sku_spec = dict(tier=dict(type='str', choices=['standard', 'premium'], required=True), family=dict(type='str', choices=['unlimiteddata', 'metereddata'], required=True))
        self.authorizations_spec = dict(name=dict(type='str', required=True))
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), location=dict(type='str'), sku=dict(type='dict', options=self.sku_spec), allow_classic_operations=dict(type='bool'), authorizations=dict(type='list', options=self.authorizations_spec, elements='dict'), state=dict(choices=['present', 'absent'], default='present', type='str'), service_provider_properties=dict(type='dict', options=self.service_provider_properties_spec), global_reach_enabled=dict(type='bool'))
        self.resource_group = None
        self.name = None
        self.location = None
        self.allow_classic_operations = None
        self.authorizations = None
        self.service_provider_properties = None
        self.global_reach_enabled = None
        self.sku = None
        self.tags = None
        self.state = None
        self.results = dict(changed=False, state=dict())
        super(AzureExpressRoute, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            setattr(self, key, kwargs[key])
        self.results['check_mode'] = self.check_mode
        resource_group = self.get_resource_group(self.resource_group)
        results = dict()
        changed = False
        try:
            self.log('Fetching Express Route Circuits {0}'.format(self.name))
            express_route_circuit = self.network_client.express_route_circuits.get(self.resource_group, self.name)
            results = express_route_to_dict(express_route_circuit)
            if self.state == 'present':
                changed = False
                update_tags, results['tags'] = self.update_tags(results['tags'])
                if update_tags:
                    changed = True
            elif self.state == 'absent':
                changed = True
        except ResourceNotFoundError:
            if self.state == 'present':
                changed = True
            else:
                changed = False
        self.results['changed'] = changed
        self.results['state'] = results
        if self.check_mode:
            return self.results
        if changed:
            if self.state == 'present':
                self.results['state'] = self.create_or_update_express_route(self.module.params)
            elif self.state == 'absent':
                self.delete_expressroute()
                self.results['state']['status'] = 'Deleted'
        return self.results

    def create_or_update_express_route(self, params):
        """
        Create or update Express route.
        :return: create or update Express route instance state dictionary
        """
        self.log('create or update Express Route {0}'.format(self.name))
        try:
            params['sku']['name'] = params.get('sku').get('tier') + '_' + params.get('sku').get('family')
            poller = self.network_client.express_route_circuits.begin_create_or_update(resource_group_name=params.get('resource_group'), circuit_name=params.get('name'), parameters=params)
            result = self.get_poller_result(poller)
            self.log('Response : {0}'.format(result))
        except Exception as ex:
            self.fail('Failed to create express route {0} in resource group {1}: {2}'.format(self.name, self.resource_group, str(ex)))
        return express_route_to_dict(result)

    def delete_expressroute(self):
        """
        Deletes specified express route circuit
        :return True
        """
        self.log('Deleting the express route {0}'.format(self.name))
        try:
            poller = self.network_client.express_route_circuits.begin_delete(self.resource_group, self.name)
            result = self.get_poller_result(poller)
        except Exception as e:
            self.log('Error attempting to delete express route.')
            self.fail('Error deleting the express route : {0}'.format(str(e)))
        return result