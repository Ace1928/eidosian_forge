from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def StopAsyncReplication(self, request, global_params=None):
    """Stops asynchronous replication. Can be invoked either on the primary or on the secondary disk.

      Args:
        request: (ComputeRegionDisksStopAsyncReplicationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('StopAsyncReplication')
    return self._RunMethod(config, request, global_params=global_params)