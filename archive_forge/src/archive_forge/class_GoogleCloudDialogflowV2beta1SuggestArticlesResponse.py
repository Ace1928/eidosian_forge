from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1SuggestArticlesResponse(_messages.Message):
    """The response message for Participants.SuggestArticles.

  Fields:
    articleAnswers: Output only. Articles ordered by score in descending
      order.
    contextSize: Number of messages prior to and including latest_message to
      compile the suggestion. It may be smaller than the
      SuggestArticlesResponse.context_size field in the request if there
      aren't that many messages in the conversation.
    latestMessage: The name of the latest conversation message used to compile
      suggestion for. Format: `projects//locations//conversations//messages/`.
  """
    articleAnswers = _messages.MessageField('GoogleCloudDialogflowV2beta1ArticleAnswer', 1, repeated=True)
    contextSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    latestMessage = _messages.StringField(3)