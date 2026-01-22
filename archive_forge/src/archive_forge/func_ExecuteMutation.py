from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.firebasedataconnect.v1alpha import firebasedataconnect_v1alpha_messages as messages
def ExecuteMutation(self, request, global_params=None):
    """Execute a predefined mutation in a Connector.

      Args:
        request: (FirebasedataconnectProjectsLocationsServicesConnectorsExecuteMutationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ExecuteMutationResponse) The response message.
      """
    config = self.GetMethodConfig('ExecuteMutation')
    return self._RunMethod(config, request, global_params=global_params)