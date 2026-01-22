from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.pubsublite.v1 import pubsublite_v1_messages as messages
def ComputeMessageStats(self, request, global_params=None):
    """Compute statistics about a range of messages in a given topic and partition.

      Args:
        request: (PubsubliteTopicStatsProjectsLocationsTopicsComputeMessageStatsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ComputeMessageStatsResponse) The response message.
      """
    config = self.GetMethodConfig('ComputeMessageStats')
    return self._RunMethod(config, request, global_params=global_params)