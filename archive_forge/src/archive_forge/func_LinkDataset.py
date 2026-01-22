from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.storageinsights.v1 import storageinsights_v1_messages as messages
def LinkDataset(self, request, global_params=None):
    """LinkDataset method for the projects_locations_datasetConfigs service.

      Args:
        request: (StorageinsightsProjectsLocationsDatasetConfigsLinkDatasetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('LinkDataset')
    return self._RunMethod(config, request, global_params=global_params)