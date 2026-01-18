from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def update_runbook(self, parameters):
    try:
        response = self.automation_client.runbook.update(self.resource_group, self.automation_account_name, self.name, parameters)
        return self.to_dict(response)
    except Exception as exc:
        self.fail('Error when updating automation account {0}: {1}'.format(self.name, exc.message))