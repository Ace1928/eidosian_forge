from __future__ import absolute_import, division, print_function
import time
import json
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_rest import GenericRestClient
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
class AzureRMManagementGroups(AzureRMModuleBaseExt):

    def __init__(self):
        self.module_arg_spec = dict(group_id=dict(type='str', updatable=False, required=True), name=dict(type='str', updatable=False), id=dict(type='str'), type=dict(type='str'), properties=dict(type='dict', disposition='/', options=dict(tenant_id=dict(type='str', disposition='tenantId'), display_name=dict(type='str', disposition='displayName'), parent_id=dict(type='str', disposition='details/parent/id'))), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.group_id = None
        self.state = None
        self.results = dict(changed=False)
        self.mgmt_client = None
        self.url = None
        self.status_code = [200, 201, 202]
        self.to_do = Actions.NoAction
        self.body = {}
        self.query_parameters = {}
        self.query_parameters['api-version'] = '2018-03-01-preview'
        self.header_parameters = {}
        self.header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        super(AzureRMManagementGroups, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()):
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            elif kwargs[key] is not None:
                self.body[key] = kwargs[key]
        self.inflate_parameters(self.module_arg_spec, self.body, 0)
        old_response = None
        response = None
        self.mgmt_client = self.get_mgmt_svc_client(GenericRestClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        self.url = '/providers' + '/Microsoft.Management' + '/managementGroups' + '/{{ management_group_name }}'
        self.url = self.url.replace('{{ management_group_name }}', self.group_id)
        old_response = self.get_resource()
        if not old_response:
            self.log("ManagementGroup instance doesn't exist")
            if self.state == 'absent':
                self.log("Old instance didn't exist")
            else:
                self.to_do = Actions.Create
        else:
            self.log('ManagementGroup instance already exists')
            if self.state == 'absent':
                self.to_do = Actions.Delete
            else:
                modifiers = {}
                self.results['compare'] = []
                self.create_compare_modifiers(self.module_arg_spec, '', modifiers)
                self.results['modifiers'] = modifiers
                if not self.default_compare(modifiers, self.body, old_response, '', self.results):
                    self.to_do = Actions.Update
        if self.to_do == Actions.Create or self.to_do == Actions.Update:
            self.log('Need to Create / Update the ManagementGroup instance')
            if self.check_mode:
                self.results['changed'] = True
                return self.results
            response = self.create_update_resource()
            self.results['body'] = self.body
            self.results['changed'] = True
            self.log('Creation / Update done')
        elif self.to_do == Actions.Delete:
            self.log('ManagementGroup instance deleted')
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            self.delete_resource()
            while self.get_resource():
                time.sleep(20)
        else:
            self.log('ManagementGroup instance unchanged')
            self.results['changed'] = False
            response = old_response
        if response:
            self.results['id'] = response['id']
            self.results['type'] = response['type']
            self.results['name'] = response['name']
            self.results['properties'] = response['properties']
        return self.results

    def create_update_resource(self):
        try:
            response = self.mgmt_client.query(self.url, 'PUT', self.query_parameters, self.header_parameters, self.body, self.status_code, 600, 30)
        except Exception as exc:
            self.log('Error attempting to create the ManagementGroup instance.')
            self.fail('Error creating the ManagementGroup instance: {0}'.format(str(exc)))
        try:
            response = json.loads(response.body())
        except Exception:
            response = {'text': response.context['deserialized_data']}
            pass
        return response

    def delete_resource(self):
        try:
            response = self.mgmt_client.query(self.url, 'DELETE', self.query_parameters, self.header_parameters, None, self.status_code, 600, 30)
        except Exception as e:
            self.log('Error attempting to delete the ManagementGroup instance.')
            self.fail('Error deleting the ManagementGroup instance: {0}'.format(str(e)))
        return True

    def get_resource(self):
        found = False
        try:
            response = self.mgmt_client.query(self.url, 'GET', self.query_parameters, self.header_parameters, None, self.status_code, 600, 30)
            found = True
            response = json.loads(response.body())
            self.log('Response : {0}'.format(response))
        except Exception as e:
            self.log('Did not find the ManagementGroup instance. msg: {0}'.format(e))
        if found is True:
            return response
        return False