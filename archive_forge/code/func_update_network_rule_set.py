from __future__ import absolute_import, division, print_function
import copy
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AZURE_SUCCESS_STATE, AzureRMModuleBase
from ansible.module_utils._text import to_native
def update_network_rule_set(self):
    if not self.check_mode:
        try:
            parameters = self.storage_models.StorageAccountUpdateParameters(network_rule_set=self.network_acls)
            self.storage_client.storage_accounts.update(self.resource_group, self.name, parameters)
        except Exception as exc:
            self.fail('Failed to update account type: {0}'.format(str(exc)))