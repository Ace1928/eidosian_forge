from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datafusion.v1beta1 import datafusion_v1beta1_messages as messages
def Restart(self, request, global_params=None):
    """Restart a single Data Fusion instance. At the end of an operation instance is fully restarted.

      Args:
        request: (DatafusionProjectsLocationsInstancesRestartRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('Restart')
    return self._RunMethod(config, request, global_params=global_params)