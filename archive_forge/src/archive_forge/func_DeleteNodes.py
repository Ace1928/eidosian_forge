from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def DeleteNodes(self, request, global_params=None):
    """Deletes specified nodes from the node group.

      Args:
        request: (ComputeNodeGroupsDeleteNodesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('DeleteNodes')
    return self._RunMethod(config, request, global_params=global_params)