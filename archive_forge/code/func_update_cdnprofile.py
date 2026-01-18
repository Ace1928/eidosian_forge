from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
import uuid
def update_cdnprofile(self):
    """
        Updates a Azure CDN profile.

        :return: deserialized Azure CDN profile instance state dictionary
        """
    self.log('Updating the Azure CDN profile instance {0}'.format(self.name))
    try:
        poller = self.cdn_client.profiles.begin_update(self.resource_group, self.name, {'tags': self.tags})
        response = self.get_poller_result(poller)
        return cdnprofile_to_dict(response)
    except Exception as exc:
        self.log('Error attempting to update Azure CDN profile instance.')
        self.fail('Error updating Azure CDN profile instance: {0}'.format(exc.message))