from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2ListMessagesResponse(_messages.Message):
    """The response message for Conversations.ListMessages.

  Fields:
    messages: The list of messages. There will be a maximum number of items
      returned based on the page_size field in the request. `messages` is
      sorted by `create_time` in descending order.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
  """
    messages = _messages.MessageField('GoogleCloudDialogflowV2Message', 1, repeated=True)
    nextPageToken = _messages.StringField(2)