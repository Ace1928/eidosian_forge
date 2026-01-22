from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBase
class AzureRMRegistrationDefinitionInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(scope=dict(type='str'), registration_definition_id=dict(type='str'))
        self.scope = None
        self.registration_definition_id = None
        self.results = dict(changed=False)
        self.mgmt_client = None
        self.state = None
        self.url = None
        self.status_code = [200]
        self.mgmt_client = None
        super(AzureRMRegistrationDefinitionInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        if not self.scope:
            self.scope = '/subscriptions/' + self.subscription_id
        else:
            self.scope = '/subscriptions/' + self.scope
        self.mgmt_client = self.get_mgmt_svc_client(ManagedServicesClient, base_url=self._cloud_environment.endpoints.resource_manager, api_version='2019-09-01', is_track2=True, suppress_subscription_id=True)
        if self.registration_definition_id is not None:
            self.results['registration_definitions'] = self.format_item(self.get())
        else:
            self.results['registration_definitions'] = self.format_item(self.list())
        return self.results

    def get(self):
        response = None
        try:
            response = self.mgmt_client.registration_definitions.get(scope=self.scope, registration_definition_id=self.registration_definition_id)
        except Exception as e:
            self.log('Could not get info for @(Model.ModuleOperationNameUpper).')
        return response

    def list(self):
        response = None
        try:
            response = self.mgmt_client.registration_definitions.list(scope=self.scope)
        except Exception as e:
            self.log('Could not get info for @(Model.ModuleOperationNameUpper).')
        return response

    def format_item(self, item):
        if hasattr(item, 'as_dict'):
            return [item.as_dict()]
        else:
            result = []
            items = list(item)
            for tmp in items:
                result.append(tmp.as_dict())
            return result