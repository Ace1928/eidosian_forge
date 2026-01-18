from __future__ import absolute_import, division, print_function
import copy
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AZURE_SUCCESS_STATE, AzureRMModuleBase
from ansible.module_utils._text import to_native
def update_static_website(self):
    if self.kind == 'FileStorage':
        return
    try:
        self.get_blob_service_client(self.resource_group, self.name).set_service_properties(static_website=self.static_website)
    except Exception as exc:
        self.fail('Failed to set static website config: {0}'.format(str(exc)))