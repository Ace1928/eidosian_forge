from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2HumanAgentAssistantConfigConversationModelConfig(_messages.Message):
    """Custom conversation models used in agent assist feature. Supported
  feature: ARTICLE_SUGGESTION, SMART_COMPOSE, SMART_REPLY,
  CONVERSATION_SUMMARIZATION.

  Fields:
    baselineModelVersion: Version of current baseline model. It will be
      ignored if model is set. Valid versions are: Article Suggestion baseline
      model: - 0.9 - 1.0 (default) Summarization baseline model: - 1.0
    model: Conversation model resource name. Format:
      `projects//conversationModels/`.
  """
    baselineModelVersion = _messages.StringField(1)
    model = _messages.StringField(2)