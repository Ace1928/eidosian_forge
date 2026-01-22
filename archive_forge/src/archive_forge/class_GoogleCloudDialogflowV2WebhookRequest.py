from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2WebhookRequest(_messages.Message):
    """The request message for a webhook call.

  Fields:
    originalDetectIntentRequest: Optional. The contents of the original
      request that was passed to `[Streaming]DetectIntent` call.
    queryResult: The result of the conversational query or event processing.
      Contains the same value as
      `[Streaming]DetectIntentResponse.query_result`.
    responseId: The unique identifier of the response. Contains the same value
      as `[Streaming]DetectIntentResponse.response_id`.
    session: The unique identifier of detectIntent request session. Can be
      used to identify end-user inside webhook implementation. Format:
      `projects//agent/sessions/`, or
      `projects//agent/environments//users//sessions/`.
  """
    originalDetectIntentRequest = _messages.MessageField('GoogleCloudDialogflowV2OriginalDetectIntentRequest', 1)
    queryResult = _messages.MessageField('GoogleCloudDialogflowV2QueryResult', 2)
    responseId = _messages.StringField(3)
    session = _messages.StringField(4)