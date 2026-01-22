from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
from ansible.module_utils.common.dict_transformations import _snake_to_camel
class AzureRMBatchAccount(AzureRMModuleBaseExt):
    """Configuration class for an Azure RM Batch Account resource"""

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(required=True, type='str'), name=dict(required=True, type='str'), location=dict(type='str', updatable=False, disposition='/'), auto_storage_account=dict(type='raw'), key_vault=dict(type='raw', no_log=True, updatable=False, disposition='/'), pool_allocation_mode=dict(default='batch_service', type='str', choices=['batch_service', 'user_subscription'], updatable=False, disposition='/'), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.resource_group = None
        self.name = None
        self.batch_account = dict()
        self.results = dict(changed=False)
        self.mgmt_client = None
        self.state = None
        self.to_do = Actions.NoAction
        super(AzureRMBatchAccount, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        """Main module execution method"""
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            elif kwargs[key] is not None:
                self.batch_account[key] = kwargs[key]
        resource_group = self.get_resource_group(self.resource_group)
        if self.batch_account.get('location') is None:
            self.batch_account['location'] = resource_group.location
        if self.batch_account.get('auto_storage_account') is not None:
            self.batch_account['auto_storage'] = {'storage_account_id': self.normalize_resource_id(self.batch_account.pop('auto_storage_account'), '/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.Storage/storageAccounts/{name}')}
        if self.batch_account.get('key_vault') is not None:
            id = self.normalize_resource_id(self.batch_account.pop('key_vault'), '/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.KeyVault/vaults/{name}')
            url = 'https://' + id.split('/').pop() + '.vault.azure.net/'
            self.batch_account['key_vault_reference'] = {'id': id, 'url': url}
        self.batch_account['pool_allocation_mode'] = _snake_to_camel(self.batch_account['pool_allocation_mode'], True)
        response = None
        self.mgmt_client = self.get_mgmt_svc_client(BatchManagementClient, base_url=self._cloud_environment.endpoints.resource_manager, is_track2=True)
        old_response = self.get_batchaccount()
        if not old_response:
            self.log("Batch Account instance doesn't exist")
            if self.state == 'absent':
                self.log("Old instance didn't exist")
            else:
                self.to_do = Actions.Create
        else:
            self.log('Batch Account instance already exists')
            if self.state == 'absent':
                self.to_do = Actions.Delete
            elif self.state == 'present':
                self.results['old'] = old_response
                self.results['new'] = self.batch_account
                update_tags, self.tags = self.update_tags(old_response['tags'])
                if self.batch_account.get('auto_storage_account') is not None:
                    if old_response['auto_storage']['storage_account_id'] != self.batch_account['auto_storage']['storage_account_id']:
                        self.to_do = Actions.Update
                if update_tags:
                    self.to_do = Actions.Update
        if self.to_do == Actions.Create or self.to_do == Actions.Update:
            self.log('Need to Create / Update the Batch Account instance')
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            response = self.create_update_batchaccount()
            self.log('Creation / Update done')
        elif self.to_do == Actions.Delete:
            self.log('Batch Account instance deleted')
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            self.delete_batchaccount()
        else:
            self.log('Batch Account instance unchanged')
            self.results['changed'] = False
            response = old_response
        if self.state == 'present':
            self.results.update({'id': response.get('id', None), 'account_endpoint': response.get('account_endpoint', None)})
        return self.results

    def create_update_batchaccount(self):
        """
        Creates or updates Batch Account with the specified configuration.

        :return: deserialized Batch Account instance state dictionary
        """
        self.log('Creating / Updating the Batch Account instance {0}'.format(self.name))
        try:
            if self.to_do == Actions.Create:
                response = self.mgmt_client.batch_account.begin_create(resource_group_name=self.resource_group, account_name=self.name, parameters=self.batch_account)
            else:
                response = self.mgmt_client.batch_account.update(resource_group_name=self.resource_group, account_name=self.name, parameters=dict(tags=self.tags, auto_storage=self.batch_account.get('self.batch_account')))
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        except Exception as exc:
            self.log('Error attempting to create the Batch Account instance.')
            self.fail('Error creating the Batch Account instance: {0}'.format(str(exc)))
        return response.as_dict()

    def delete_batchaccount(self):
        """
        Deletes specified Batch Account instance in the specified subscription and resource group.

        :return: True
        """
        self.log('Deleting the Batch Account instance {0}'.format(self.name))
        try:
            response = self.mgmt_client.batch_account.begin_delete(resource_group_name=self.resource_group, account_name=self.name)
        except Exception as e:
            self.log('Error attempting to delete the Batch Account instance.')
            self.fail('Error deleting the Batch Account instance: {0}'.format(str(e)))
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
        return True

    def get_batchaccount(self):
        """
        Gets the properties of the specified Batch Account
        :return: deserialized Batch Account instance state dictionary
        """
        self.log('Checking if the Batch Account instance {0} is present'.format(self.name))
        found = False
        try:
            response = self.mgmt_client.batch_account.get(resource_group_name=self.resource_group, account_name=self.name)
            found = True
            self.log('Response : {0}'.format(response))
            self.log('Batch Account instance : {0} found'.format(response.name))
        except ResourceNotFoundError as e:
            self.log('Did not find the Batch Account instance. Exception as {0}'.format(e))
        if found is True:
            return self.format_item(response.as_dict())
        return False

    def format_item(self, item):
        result = {'id': item['id'], 'name': item['name'], 'type': item['type'], 'location': item['location'], 'account_endpoint': item['account_endpoint'], 'provisioning_state': item['provisioning_state'], 'pool_allocation_mode': item['pool_allocation_mode'], 'auto_storage': item['auto_storage'], 'dedicated_core_quota': item['dedicated_core_quota'], 'low_priority_core_quota': item['low_priority_core_quota'], 'pool_quota': item['pool_quota'], 'active_job_and_job_schedule_quota': item['active_job_and_job_schedule_quota'], 'tags': item.get('tags')}
        return result