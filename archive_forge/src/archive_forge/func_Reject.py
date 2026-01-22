from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.telcoautomation.v1 import telcoautomation_v1_messages as messages
def Reject(self, request, global_params=None):
    """Rejects a blueprint revision proposal and flips it back to Draft state.

      Args:
        request: (TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsRejectRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Blueprint) The response message.
      """
    config = self.GetMethodConfig('Reject')
    return self._RunMethod(config, request, global_params=global_params)