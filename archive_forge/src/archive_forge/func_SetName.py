from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def SetName(self, request, global_params=None):
    """Sets name of an instance.

      Args:
        request: (ComputeInstancesSetNameRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('SetName')
    return self._RunMethod(config, request, global_params=global_params)