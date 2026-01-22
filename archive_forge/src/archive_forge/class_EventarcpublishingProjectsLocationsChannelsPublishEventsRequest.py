from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EventarcpublishingProjectsLocationsChannelsPublishEventsRequest(_messages.Message):
    """A EventarcpublishingProjectsLocationsChannelsPublishEventsRequest
  object.

  Fields:
    channel: The full name of the channel to publish to. For example:
      `projects/{project}/locations/{location}/channels/{channel-id}`.
    googleCloudEventarcPublishingV1PublishEventsRequest: A
      GoogleCloudEventarcPublishingV1PublishEventsRequest resource to be
      passed as the request body.
  """
    channel = _messages.StringField(1, required=True)
    googleCloudEventarcPublishingV1PublishEventsRequest = _messages.MessageField('GoogleCloudEventarcPublishingV1PublishEventsRequest', 2)