from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2HumanAgentAssistantConfigConversationProcessConfig(_messages.Message):
    """Config to process conversation.

  Fields:
    recentSentencesCount: Number of recent non-small-talk sentences to use as
      context for article and FAQ suggestion
  """
    recentSentencesCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)