from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_rest import GenericRestClient
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
import json
class BackupPolicyVMInfo(AzureRMModuleBaseExt):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), vault_name=dict(type='str', required=True))
        self.resource_group = None
        self.name = None
        self.vault_name = None
        self.results = dict(changed=False)
        self.mgmt_client = None
        self.url = None
        self.status_code = [200, 202]
        self.query_parameters = {}
        self.query_parameters['api-version'] = '2019-05-13'
        self.header_parameters = {}
        self.header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        super(BackupPolicyVMInfo, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def get_url(self):
        return '/subscriptions/' + self.subscription_id + '/resourceGroups/' + self.resource_group + '/providers/Microsoft.RecoveryServices' + '/vaults' + '/' + self.vault_name + '/' + 'backupPolicies/' + self.name

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()):
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
        self.url = self.get_url()
        self.mgmt_client = self.get_mgmt_svc_client(GenericRestClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        response = self.get_resource()
        changed = False
        self.results['response'] = response
        self.results['changed'] = changed
        return self.results

    def get_resource(self):
        try:
            response = self.mgmt_client.query(self.url, 'GET', self.query_parameters, self.header_parameters, None, self.status_code, 600, 30)
            found = True
        except Exception as e:
            self.log('Backup policy does not exist.')
            self.fail('Error in fetching VM Backup Policy {0}'.format(str(e)))
        try:
            response = json.loads(response.body())
        except Exception:
            response = {'text': response.context['deserialized_data']}
        return response