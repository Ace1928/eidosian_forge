from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def GetHealth(self, request, global_params=None):
    """Gets the most recent health check results for each IP for the instance that is referenced by the given target pool.

      Args:
        request: (ComputeTargetPoolsGetHealthRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TargetPoolInstanceHealth) The response message.
      """
    config = self.GetMethodConfig('GetHealth')
    return self._RunMethod(config, request, global_params=global_params)