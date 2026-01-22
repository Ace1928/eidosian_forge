from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networksecurity.v1 import networksecurity_v1_messages as messages
def RemoveItems(self, request, global_params=None):
    """Removes items from an address group.

      Args:
        request: (NetworksecurityProjectsLocationsAddressGroupsRemoveItemsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('RemoveItems')
    return self._RunMethod(config, request, global_params=global_params)