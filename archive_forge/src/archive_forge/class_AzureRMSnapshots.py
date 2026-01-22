from __future__ import absolute_import, division, print_function
import time
import json
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_rest import GenericRestClient
class AzureRMSnapshots(AzureRMModuleBaseExt):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', updatable=False, disposition='resourceGroupName', required=True), name=dict(type='str', updatable=False, disposition='snapshotName', required=True), location=dict(type='str', updatable=False, disposition='/'), sku=dict(type='dict', disposition='/', options=dict(name=dict(type='str', choices=['Standard_LRS', 'Premium_LRS', 'Standard_ZRS']), tier=dict(type='str'))), os_type=dict(type='str', disposition='/properties/osType', choices=['Windows', 'Linux']), incremental=dict(type='bool', default=False), creation_data=dict(type='dict', disposition='/properties/creationData', options=dict(create_option=dict(type='str', disposition='createOption', choices=['Import', 'Copy']), source_uri=dict(type='str', disposition='sourceUri', purgeIfNone=True), source_id=dict(type='str', disposition='sourceResourceId', purgeIfNone=True))), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.resource_group = None
        self.name = None
        self.id = None
        self.name = None
        self.type = None
        self.managed_by = None
        self.results = dict(changed=False)
        self.mgmt_client = None
        self.state = None
        self.url = None
        self.status_code = [200, 201, 202]
        self.to_do = Actions.NoAction
        self.body = {}
        self.body['properties'] = dict()
        self.query_parameters = {}
        self.query_parameters['api-version'] = '2019-03-01'
        self.header_parameters = {}
        self.header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        super(AzureRMSnapshots, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()):
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            elif kwargs[key] is not None:
                if key == 'incremental':
                    self.body['properties']['incremental'] = kwargs[key]
                else:
                    self.body[key] = kwargs[key]
        self.inflate_parameters(self.module_arg_spec, self.body, 0)
        old_response = None
        response = None
        self.mgmt_client = self.get_mgmt_svc_client(GenericRestClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        resource_group = self.get_resource_group(self.resource_group)
        if 'location' not in self.body:
            self.body['location'] = resource_group.location
        self.url = '/subscriptions' + '/{{ subscription_id }}' + '/resourceGroups' + '/{{ resource_group }}' + '/providers' + '/Microsoft.Compute' + '/snapshots' + '/{{ snapshot_name }}'
        self.url = self.url.replace('{{ subscription_id }}', self.subscription_id)
        self.url = self.url.replace('{{ resource_group }}', self.resource_group)
        self.url = self.url.replace('{{ snapshot_name }}', self.name)
        old_response = self.get_resource()
        if not old_response:
            self.log("Snapshot instance doesn't exist")
            if self.state == 'absent':
                self.log("Old instance didn't exist")
            else:
                self.to_do = Actions.Create
        else:
            self.log('Snapshot instance already exists')
            if self.state == 'absent':
                self.to_do = Actions.Delete
            else:
                modifiers = {}
                self.create_compare_modifiers(self.module_arg_spec, '', modifiers)
                self.results['modifiers'] = modifiers
                self.results['compare'] = []
                self.create_compare_modifiers(self.module_arg_spec, '', modifiers)
                if not self.default_compare(modifiers, self.body, old_response, '', self.results):
                    self.to_do = Actions.Update
        if self.to_do == Actions.Create or self.to_do == Actions.Update:
            self.log('Need to Create / Update the Snapshot instance')
            if self.check_mode:
                self.results['changed'] = True
                return self.results
            response = self.create_update_resource()
            self.results['changed'] = True
            self.log('Creation / Update done')
        elif self.to_do == Actions.Delete:
            self.log('Snapshot instance deleted')
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            self.delete_resource()
            while self.get_resource():
                time.sleep(20)
        else:
            self.log('Snapshot instance unchanged')
            self.results['changed'] = False
            response = old_response
        if response:
            self.results['id'] = response['id']
        return self.results

    def create_update_resource(self):
        response = None
        try:
            response = self.mgmt_client.query(url=self.url, method='PUT', query_parameters=self.query_parameters, header_parameters=self.header_parameters, body=self.body, expected_status_codes=self.status_code, polling_timeout=600, polling_interval=30)
        except Exception as exc:
            self.log('Error attempting to create the Snapshot instance.')
            self.fail('Error creating the Snapshot instance: {0}'.format(str(exc)))
        if hasattr(response, 'body'):
            response = json.loads(response.body())
        elif hasattr(response, 'context'):
            response = response.context['deserialized_data']
        else:
            self.fail('Create or Updating fail, no match message return, return info as {0}'.format(response))
        return response

    def delete_resource(self):
        try:
            response = self.mgmt_client.query(url=self.url, method='DELETE', query_parameters=self.query_parameters, header_parameters=self.header_parameters, body=None, expected_status_codes=self.status_code, polling_timeout=600, polling_interval=30)
        except Exception as e:
            self.log('Error attempting to delete the Snapshot instance.')
            self.fail('Error deleting the Snapshot instance: {0}'.format(str(e)))
        return True

    def get_resource(self):
        found = False
        try:
            response = self.mgmt_client.query(url=self.url, method='GET', query_parameters=self.query_parameters, header_parameters=self.header_parameters, body=None, expected_status_codes=self.status_code, polling_timeout=600, polling_interval=30)
            response = json.loads(response.body())
            found = True
            self.log('Response : {0}'.format(response))
        except Exception as e:
            self.log('Did not find the Snapshot instance.')
        if found is True:
            return response
        return False