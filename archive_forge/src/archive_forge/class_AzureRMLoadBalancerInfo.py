from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMLoadBalancerInfo(AzureRMModuleBase):
    """Utility class to get load balancer facts"""

    def __init__(self):
        self.module_args = dict(name=dict(type='str'), resource_group=dict(type='str'), tags=dict(type='list', elements='str'))
        self.results = dict(changed=False, loadbalancers=[])
        self.name = None
        self.resource_group = None
        self.tags = None
        super(AzureRMLoadBalancerInfo, self).__init__(derived_arg_spec=self.module_args, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_loadbalancer_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_loadbalancer_facts' module has been renamed to 'azure_rm_loadbalancer_info'", version=(2.9,))
        for key in self.module_args:
            setattr(self, key, kwargs[key])
        self.results['loadbalancers'] = self.get_item() if self.name else self.list_items()
        return self.results

    def get_item(self):
        """Get a single load balancer"""
        self.log('Get properties for {0}'.format(self.name))
        item = None
        result = []
        try:
            item = self.network_client.load_balancers.get(self.resource_group, self.name)
        except ResourceNotFoundError:
            pass
        if item and self.has_tags(item.tags, self.tags):
            result = [self.serialize_obj(item, AZURE_OBJECT_CLASS)]
        return result

    def list_items(self):
        """Get all load balancers"""
        self.log('List all load balancers')
        if self.resource_group:
            try:
                response = self.network_client.load_balancers.list(self.resource_group)
            except ResourceNotFoundError as exc:
                self.fail('Failed to list items in resource group {0} - {1}'.format(self.resource_group, str(exc)))
        else:
            try:
                response = self.network_client.load_balancers.list_all()
            except ResourceNotFoundError as exc:
                self.fail('Failed to list all items - {0}'.format(str(exc)))
        results = []
        for item in response:
            if self.has_tags(item.tags, self.tags):
                results.append(self.serialize_obj(item, AZURE_OBJECT_CLASS))
        return results