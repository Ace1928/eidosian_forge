from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def ApplyUpdatesToInstances(self, request, global_params=None):
    """Apply updates to selected instances the managed instance group.

      Args:
        request: (ComputeRegionInstanceGroupManagersApplyUpdatesToInstancesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('ApplyUpdatesToInstances')
    return self._RunMethod(config, request, global_params=global_params)