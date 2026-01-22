from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2SuggestFaqAnswersResponse(_messages.Message):
    """The request message for Participants.SuggestFaqAnswers.

  Fields:
    contextSize: Number of messages prior to and including latest_message to
      compile the suggestion. It may be smaller than the
      SuggestFaqAnswersRequest.context_size field in the request if there
      aren't that many messages in the conversation.
    faqAnswers: Answers extracted from FAQ documents.
    latestMessage: The name of the latest conversation message used to compile
      suggestion for. Format: `projects//locations//conversations//messages/`.
  """
    contextSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    faqAnswers = _messages.MessageField('GoogleCloudDialogflowV2FaqAnswer', 2, repeated=True)
    latestMessage = _messages.StringField(3)