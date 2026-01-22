from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListSSEGatewaysResponse(_messages.Message):
    """Message for response to listing SSEGateways

  Fields:
    nextPageToken: A token identifying a page of results the server should
      return.
    sseGateways: The list of SSEGateway
    unreachable: Locations that could not be reached.
  """
    nextPageToken = _messages.StringField(1)
    sseGateways = _messages.MessageField('SSEGateway', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)