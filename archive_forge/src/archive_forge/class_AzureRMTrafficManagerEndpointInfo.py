from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import (
class AzureRMTrafficManagerEndpointInfo(AzureRMModuleBase):
    """Utility class to get Azure Traffic Manager Endpoint facts"""

    def __init__(self):
        self.module_args = dict(profile_name=dict(type='str', required=True), resource_group=dict(type='str', required=True), name=dict(type='str'), type=dict(type='str', choices=['azure_endpoints', 'external_endpoints', 'nested_endpoints']))
        self.results = dict(changed=False, endpoints=[])
        self.profile_name = None
        self.name = None
        self.resource_group = None
        self.type = None
        super(AzureRMTrafficManagerEndpointInfo, self).__init__(derived_arg_spec=self.module_args, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_trafficmanagerendpoint_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_trafficmanagerendpoint_facts' module has been renamed to 'azure_rm_trafficmanagerendpoint_info'", version=(2.9,))
        for key in self.module_args:
            setattr(self, key, kwargs[key])
        if self.type:
            self.type = _snake_to_camel(self.type)
        if self.name and (not self.resource_group):
            self.fail('Parameter error: resource group required when filtering by name.')
        if self.name:
            self.results['endpoints'] = self.get_item()
        elif self.type:
            self.results['endpoints'] = self.list_by_type()
        else:
            self.results['endpoints'] = self.list_by_profile()
        return self.results

    def get_item(self):
        """Get a single Azure Traffic Manager endpoint"""
        self.log('Get properties for {0}'.format(self.name))
        item = None
        result = []
        try:
            item = self.traffic_manager_management_client.endpoints.get(self.resource_group, self.profile_name, self.type, self.name)
        except ResourceNotFoundError:
            pass
        if item:
            if self.type and self.type == item.type or self.type is None:
                result = [self.serialize_tm(item)]
        return result

    def list_by_profile(self):
        """Get all Azure Traffic Manager endpoints of a profile"""
        self.log('List all endpoints belongs to a Traffic Manager profile')
        try:
            response = self.traffic_manager_management_client.profiles.get(self.resource_group, self.profile_name)
        except Exception as exc:
            self.fail('Failed to list all items - {0}'.format(str(exc)))
        results = []
        if response and response.endpoints:
            for endpoint in response.endpoints:
                results.append(serialize_endpoint(endpoint, self.resource_group))
        return results

    def list_by_type(self):
        """Get all Azure Traffic Managers endpoints of a profile by type"""
        self.log('List all Traffic Manager endpoints of a profile by type')
        try:
            response = self.traffic_manager_management_client.profiles.get(self.resource_group, self.profile_name)
        except Exception as exc:
            self.fail('Failed to list all items - {0}'.format(str(exc)))
        results = []
        for item in response:
            if item.endpoints:
                for endpoint in item.endpoints:
                    if endpoint.type == self.type:
                        results.append(serialize_endpoint(endpoint, self.resource_group))
        return results