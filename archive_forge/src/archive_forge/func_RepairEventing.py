from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.connectors.v1 import connectors_v1_messages as messages
def RepairEventing(self, request, global_params=None):
    """RepaiEventing tries to repair eventing related event subscriptions.

      Args:
        request: (ConnectorsProjectsLocationsConnectionsRepairEventingRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('RepairEventing')
    return self._RunMethod(config, request, global_params=global_params)