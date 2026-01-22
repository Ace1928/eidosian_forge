from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListNotificationChannelDescriptorsResponse(_messages.Message):
    """The ListNotificationChannelDescriptors response.

  Fields:
    channelDescriptors: The monitored resource descriptors supported for the
      specified project, optionally filtered.
    nextPageToken: If not empty, indicates that there may be more results that
      match the request. Use the value in the page_token field in a subsequent
      request to fetch the next set of results. If empty, all results have
      been returned.
  """
    channelDescriptors = _messages.MessageField('NotificationChannelDescriptor', 1, repeated=True)
    nextPageToken = _messages.StringField(2)