from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1HumanAgentAssistantEvent(_messages.Message):
    """Output only. Represents a notification sent to Pub/Sub subscribers for
  agent assistant events in a specific conversation.

  Fields:
    conversation: The conversation this notification refers to. Format:
      `projects//conversations/`.
    participant: The participant that the suggestion is compiled for. And This
      field is used to call Participants.ListSuggestions API. Format:
      `projects//conversations//participants/`. It will not be set in legacy
      workflow. HumanAgentAssistantConfig.name for more information.
    suggestionResults: The suggestion results payload that this notification
      refers to. It will only be set when
      HumanAgentAssistantConfig.SuggestionConfig.group_suggestion_responses
      sets to true.
  """
    conversation = _messages.StringField(1)
    participant = _messages.StringField(2)
    suggestionResults = _messages.MessageField('GoogleCloudDialogflowV2beta1SuggestionResult', 3, repeated=True)