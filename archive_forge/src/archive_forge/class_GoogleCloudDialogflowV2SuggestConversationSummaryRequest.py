from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2SuggestConversationSummaryRequest(_messages.Message):
    """The request message for Conversations.SuggestConversationSummary.

  Fields:
    assistQueryParams: Parameters for a human assist query. Only used for
      POC/demo purpose.
    contextSize: Max number of messages prior to and including
      [latest_message] to use as context when compiling the suggestion. By
      default 500 and at most 1000.
    latestMessage: The name of the latest conversation message used as context
      for compiling suggestion. If empty, the latest message of the
      conversation will be used. Format:
      `projects//locations//conversations//messages/`.
  """
    assistQueryParams = _messages.MessageField('GoogleCloudDialogflowV2AssistQueryParameters', 1)
    contextSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    latestMessage = _messages.StringField(3)