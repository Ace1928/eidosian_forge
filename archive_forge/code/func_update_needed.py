from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def update_needed(self, old_response):
    """
        Define if storage file share update needed.
        :param old_response: dict with properties of the storage file share
        :return: True if update needed, else False
        """
    return self.access_tier is not None and self.access_tier != old_response.get('access_tier') or (self.quota is not None and self.quota != old_response.get('share_quota')) or (self.metadata is not None and self.metadata != old_response.get('metadata')) or (self.root_squash is not None and self.root_squash != old_response.get('root_squash')) or (self.enabled_protocols is not None and self.enabled_protocols != old_response.get('enabled_protocols'))