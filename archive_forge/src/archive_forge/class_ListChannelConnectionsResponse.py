from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListChannelConnectionsResponse(_messages.Message):
    """The response message for the `ListChannelConnections` method.

  Fields:
    channelConnections: The requested channel connections, up to the number
      specified in `page_size`.
    nextPageToken: A page token that can be sent to `ListChannelConnections`
      to request the next page. If this is empty, then there are no more
      pages.
    unreachable: Unreachable resources, if any.
  """
    channelConnections = _messages.MessageField('ChannelConnection', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)