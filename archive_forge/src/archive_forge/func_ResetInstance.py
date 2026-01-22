from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.baremetalsolution.v1 import baremetalsolution_v1_messages as messages
def ResetInstance(self, request, global_params=None):
    """Perform an ungraceful, hard reset on a machine (equivalent to shutting the power off, and then turning it back on).

      Args:
        request: (BaremetalsolutionProjectsLocationsInstancesResetInstanceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ResetInstanceResponse) The response message.
      """
    config = self.GetMethodConfig('ResetInstance')
    return self._RunMethod(config, request, global_params=global_params)