from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.connectors.v1 import connectors_v1_messages as messages
def UpdateSettings(self, request, global_params=None):
    """Update the global settings of a project.

      Args:
        request: (ConnectorsProjectsLocationsGlobalUpdateSettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('UpdateSettings')
    return self._RunMethod(config, request, global_params=global_params)