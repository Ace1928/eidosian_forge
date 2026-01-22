from __future__ import absolute_import, division, print_function
import copy
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AZURE_SUCCESS_STATE, AzureRMModuleBase
from ansible.module_utils._text import to_native
def account_has_blob_containers(self):
    """
        If there are blob containers, then there are likely VMs depending on this account and it should
        not be deleted.
        """
    if self.kind == 'FileStorage':
        return False
    self.log('Checking for existing blob containers')
    blob_service = self.get_blob_service_client(self.resource_group, self.name)
    try:
        response = blob_service.list_containers()
    except Exception:
        return False
    if len(list(response)) > 0:
        return True
    return False