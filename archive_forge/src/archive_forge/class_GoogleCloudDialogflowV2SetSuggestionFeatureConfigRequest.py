from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2SetSuggestionFeatureConfigRequest(_messages.Message):
    """The request message for ConversationProfiles.SetSuggestionFeature.

  Enums:
    ParticipantRoleValueValuesEnum: Required. The participant role to add or
      update the suggestion feature config. Only HUMAN_AGENT or END_USER can
      be used.

  Fields:
    participantRole: Required. The participant role to add or update the
      suggestion feature config. Only HUMAN_AGENT or END_USER can be used.
    suggestionFeatureConfig: Required. The suggestion feature config to add or
      update.
  """

    class ParticipantRoleValueValuesEnum(_messages.Enum):
        """Required. The participant role to add or update the suggestion feature
    config. Only HUMAN_AGENT or END_USER can be used.

    Values:
      ROLE_UNSPECIFIED: Participant role not set.
      HUMAN_AGENT: Participant is a human agent.
      AUTOMATED_AGENT: Participant is an automated agent, such as a Dialogflow
        agent.
      END_USER: Participant is an end user that has called or chatted with
        Dialogflow services.
    """
        ROLE_UNSPECIFIED = 0
        HUMAN_AGENT = 1
        AUTOMATED_AGENT = 2
        END_USER = 3
    participantRole = _messages.EnumField('ParticipantRoleValueValuesEnum', 1)
    suggestionFeatureConfig = _messages.MessageField('GoogleCloudDialogflowV2HumanAgentAssistantConfigSuggestionFeatureConfig', 2)