from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EventarcProjectsLocationsChannelsCreateRequest(_messages.Message):
    """A EventarcProjectsLocationsChannelsCreateRequest object.

  Fields:
    channel: A Channel resource to be passed as the request body.
    channelId: Required. The user-provided ID to be assigned to the channel.
    parent: Required. The parent collection in which to add this channel.
    validateOnly: Optional. If set, validate the request and preview the
      review, but do not post it.
  """
    channel = _messages.MessageField('Channel', 1)
    channelId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    validateOnly = _messages.BooleanField(4)