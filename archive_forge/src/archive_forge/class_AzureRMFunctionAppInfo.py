from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMFunctionAppInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(name=dict(type='str'), resource_group=dict(type='str', aliases=['resource_group_name']), tags=dict(type='list', elements='str'))
        self.results = dict(changed=False, ansible_info=dict(azure_functionapps=[]))
        self.name = None
        self.resource_group = None
        self.tags = None
        super(AzureRMFunctionAppInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_functionapp_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_functionapp_facts' module has been renamed to 'azure_rm_functionapp_info'", version=(2.9,))
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        if self.name and (not self.resource_group):
            self.fail('Parameter error: resource group required when filtering by name.')
        if self.name:
            self.results['ansible_info']['azure_functionapps'] = self.get_functionapp()
        elif self.resource_group:
            self.results['ansible_info']['azure_functionapps'] = self.list_resource_group()
        else:
            self.results['ansible_info']['azure_functionapps'] = self.list_all()
        return self.results

    def get_functionapp(self):
        self.log('Get properties for Function App {0}'.format(self.name))
        function_app = None
        result = []
        try:
            function_app = self.web_client.web_apps.get(resource_group_name=self.resource_group, name=self.name)
        except ResourceNotFoundError:
            pass
        if function_app and self.has_tags(function_app.tags, self.tags):
            result = function_app.as_dict()
        return [result]

    def list_resource_group(self):
        self.log('List items')
        try:
            response = self.web_client.web_apps.list_by_resource_group(resource_group_name=self.resource_group)
        except Exception as exc:
            self.fail('Error listing for resource group {0} - {1}'.format(self.resource_group, str(exc)))
        results = []
        for item in response:
            if self.has_tags(item.tags, self.tags):
                results.append(item.as_dict())
        return results

    def list_all(self):
        self.log('List all items')
        try:
            response = self.web_client.web_apps.list_by_resource_group(resource_group_name=self.resource_group)
        except Exception as exc:
            self.fail('Error listing all items - {0}'.format(str(exc)))
        results = []
        for item in response:
            if self.has_tags(item.tags, self.tags):
                results.append(item.as_dict())
        return results