from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1SuggestSmartRepliesResponse(_messages.Message):
    """The response message for Participants.SuggestSmartReplies.

  Fields:
    contextSize: Number of messages prior to and including latest_message to
      compile the suggestion. It may be smaller than the
      SuggestSmartRepliesRequest.context_size field in the request if there
      aren't that many messages in the conversation.
    latestMessage: The name of the latest conversation message used to compile
      suggestion for. Format: `projects//locations//conversations//messages/`.
    smartReplyAnswers: Output only. Multiple reply options provided by smart
      reply service. The order is based on the rank of the model prediction.
      The maximum number of the returned replies is set in SmartReplyConfig.
  """
    contextSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    latestMessage = _messages.StringField(2)
    smartReplyAnswers = _messages.MessageField('GoogleCloudDialogflowV2beta1SmartReplyAnswer', 3, repeated=True)