from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2HumanAgentAssistantConfig(_messages.Message):
    """Defines the Human Agent Assist to connect to a conversation.

  Fields:
    endUserSuggestionConfig: Configuration for agent assistance of end user
      participant. Currently, this feature is not general available, please
      contact Google to get access.
    humanAgentSuggestionConfig: Configuration for agent assistance of human
      agent participant.
    messageAnalysisConfig: Configuration for message analysis.
    notificationConfig: Pub/Sub topic on which to publish new agent assistant
      events.
  """
    endUserSuggestionConfig = _messages.MessageField('GoogleCloudDialogflowV2HumanAgentAssistantConfigSuggestionConfig', 1)
    humanAgentSuggestionConfig = _messages.MessageField('GoogleCloudDialogflowV2HumanAgentAssistantConfigSuggestionConfig', 2)
    messageAnalysisConfig = _messages.MessageField('GoogleCloudDialogflowV2HumanAgentAssistantConfigMessageAnalysisConfig', 3)
    notificationConfig = _messages.MessageField('GoogleCloudDialogflowV2NotificationConfig', 4)