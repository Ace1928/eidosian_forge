from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.pubsublite.v1 import pubsublite_v1_messages as messages
def ComputeHeadCursor(self, request, global_params=None):
    """Compute the head cursor for the partition. The head cursor's offset is guaranteed to be less than or equal to all messages which have not yet been acknowledged as published, and greater than the offset of any message whose publish has already been acknowledged. It is zero if there have never been messages in the partition.

      Args:
        request: (PubsubliteTopicStatsProjectsLocationsTopicsComputeHeadCursorRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ComputeHeadCursorResponse) The response message.
      """
    config = self.GetMethodConfig('ComputeHeadCursor')
    return self._RunMethod(config, request, global_params=global_params)