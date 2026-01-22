from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EventarcpublishingProjectsLocationsMessageBusesPublishRequest(_messages.Message):
    """A EventarcpublishingProjectsLocationsMessageBusesPublishRequest object.

  Fields:
    googleCloudEventarcPublishingV1PublishRequest: A
      GoogleCloudEventarcPublishingV1PublishRequest resource to be passed as
      the request body.
    messageBus: Required. The full name of the message bus to publish events
      to. Format:
      `projects/{project}/locations/{location}/messageBuses/{messageBus}`.
  """
    googleCloudEventarcPublishingV1PublishRequest = _messages.MessageField('GoogleCloudEventarcPublishingV1PublishRequest', 1)
    messageBus = _messages.StringField(2, required=True)