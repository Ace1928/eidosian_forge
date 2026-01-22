from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.notebooks.v1 import notebooks_v1_messages as messages
def UpdateMetadataItems(self, request, global_params=None):
    """Add/update metadata items for an instance.

      Args:
        request: (NotebooksProjectsLocationsInstancesUpdateMetadataItemsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (UpdateInstanceMetadataItemsResponse) The response message.
      """
    config = self.GetMethodConfig('UpdateMetadataItems')
    return self._RunMethod(config, request, global_params=global_params)