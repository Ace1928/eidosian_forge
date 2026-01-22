from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2ClearSuggestionFeatureConfigOperationMetadata(_messages.Message):
    """Metadata for a ConversationProfile.ClearSuggestionFeatureConfig
  operation.

  Enums:
    ParticipantRoleValueValuesEnum: Required. The participant role to remove
      the suggestion feature config. Only HUMAN_AGENT or END_USER can be used.
    SuggestionFeatureTypeValueValuesEnum: Required. The type of the suggestion
      feature to remove.

  Fields:
    conversationProfile: The resource name of the conversation profile.
      Format: `projects//locations//conversationProfiles/`
    createTime: Timestamp whe the request was created. The time is measured on
      server side.
    participantRole: Required. The participant role to remove the suggestion
      feature config. Only HUMAN_AGENT or END_USER can be used.
    suggestionFeatureType: Required. The type of the suggestion feature to
      remove.
  """

    class ParticipantRoleValueValuesEnum(_messages.Enum):
        """Required. The participant role to remove the suggestion feature
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

    class SuggestionFeatureTypeValueValuesEnum(_messages.Enum):
        """Required. The type of the suggestion feature to remove.

    Values:
      TYPE_UNSPECIFIED: Unspecified feature type.
      ARTICLE_SUGGESTION: Run article suggestion model for chat.
      FAQ: Run FAQ model for chat.
      SMART_REPLY: Run smart reply model for chat.
      KNOWLEDGE_SEARCH: Run knowledge search with text input from agent or
        text generated query.
    """
        TYPE_UNSPECIFIED = 0
        ARTICLE_SUGGESTION = 1
        FAQ = 2
        SMART_REPLY = 3
        KNOWLEDGE_SEARCH = 4
    conversationProfile = _messages.StringField(1)
    createTime = _messages.StringField(2)
    participantRole = _messages.EnumField('ParticipantRoleValueValuesEnum', 3)
    suggestionFeatureType = _messages.EnumField('SuggestionFeatureTypeValueValuesEnum', 4)