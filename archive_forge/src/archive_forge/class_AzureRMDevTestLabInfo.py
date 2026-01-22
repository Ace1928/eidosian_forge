from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMDevTestLabInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str'), name=dict(type='str'), tags=dict(type='list', elements='str'))
        self.results = dict(changed=False)
        self.mgmt_client = None
        self.resource_group = None
        self.name = None
        self.tags = None
        super(AzureRMDevTestLabInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_devtestlab_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_devtestlab_facts' module has been renamed to 'azure_rm_devtestlab_info'", version=(2.9,))
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        self.mgmt_client = self.get_mgmt_svc_client(DevTestLabsClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        if self.resource_group is not None:
            if self.name is not None:
                self.results['labs'] = self.get()
            else:
                self.results['labs'] = self.list_by_resource_group()
        else:
            self.results['labs'] = self.list_by_subscription()
        return self.results

    def list_by_resource_group(self):
        response = None
        results = []
        try:
            response = self.mgmt_client.labs.list_by_resource_group(resource_group_name=self.resource_group)
            self.log('Response : {0}'.format(response))
        except Exception as e:
            self.log('Could not get facts for Lab.')
        if response is not None:
            for item in response:
                if self.has_tags(item.tags, self.tags):
                    results.append(self.format_response(item))
        return results

    def list_by_subscription(self):
        response = None
        results = []
        try:
            response = self.mgmt_client.labs.list_by_subscription()
            self.log('Response : {0}'.format(response))
        except Exception as e:
            self.log('Could not get facts for Lab.')
        if response is not None:
            for item in response:
                if self.has_tags(item.tags, self.tags):
                    results.append(self.format_response(item))
        return results

    def get(self):
        response = None
        results = []
        try:
            response = self.mgmt_client.labs.get(resource_group_name=self.resource_group, name=self.name)
            self.log('Response : {0}'.format(response))
        except ResourceNotFoundError as e:
            self.log('Could not get facts for Lab.')
        if response and self.has_tags(response.tags, self.tags):
            results.append(self.format_response(response))
        return results

    def format_response(self, item):
        d = item.as_dict()
        d = {'id': d.get('id', None), 'resource_group': self.resource_group, 'name': d.get('name', None), 'location': d.get('location', '').replace(' ', '').lower(), 'storage_type': d.get('lab_storage_type', '').lower(), 'premium_data_disks': d.get('premium_data_disks') == 'Enabled', 'provisioning_state': d.get('provisioning_state'), 'artifacts_storage_account': d.get('artifacts_storage_account'), 'default_premium_storage_account': d.get('default_premium_storage_account'), 'default_storage_account': d.get('default_storage_account'), 'premium_data_disk_storage_account': d.get('premium_data_disk_storage_account'), 'vault_name': d.get('vault_name'), 'tags': d.get('tags', None)}
        return d