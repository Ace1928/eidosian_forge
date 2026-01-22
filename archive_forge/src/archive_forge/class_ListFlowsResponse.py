from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListFlowsResponse(_messages.Message):
    """Response message for EventingFlow.ListFlows.

  Fields:
    flows: The list of items.
    nextPageToken: A token to retrieve next page of results.
  """
    flows = _messages.MessageField('Flow', 1, repeated=True)
    nextPageToken = _messages.StringField(2)