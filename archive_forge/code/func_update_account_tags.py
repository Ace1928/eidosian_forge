from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def update_account_tags(self, param):
    try:
        return self.automation_client.automation_account.update(self.resource_group, self.name, param)
    except Exception as exc:
        self.fail('Error when updating automation account {0}: {1}'.format(self.name, exc.message))