from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.eventarcpublishing.v1 import eventarcpublishing_v1_messages as messages
def PublishEvents(self, request, global_params=None):
    """Publish events to a subscriber's channel.

      Args:
        request: (EventarcpublishingProjectsLocationsChannelsPublishEventsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudEventarcPublishingV1PublishEventsResponse) The response message.
      """
    config = self.GetMethodConfig('PublishEvents')
    return self._RunMethod(config, request, global_params=global_params)