from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsConversationProfilesClearSuggestionFeatureConfigRequest(_messages.Message):
    """A
  DialogflowProjectsConversationProfilesClearSuggestionFeatureConfigRequest
  object.

  Fields:
    conversationProfile: Required. The Conversation Profile to add or update
      the suggestion feature config. Format:
      `projects//locations//conversationProfiles/`.
    googleCloudDialogflowV2ClearSuggestionFeatureConfigRequest: A
      GoogleCloudDialogflowV2ClearSuggestionFeatureConfigRequest resource to
      be passed as the request body.
  """
    conversationProfile = _messages.StringField(1, required=True)
    googleCloudDialogflowV2ClearSuggestionFeatureConfigRequest = _messages.MessageField('GoogleCloudDialogflowV2ClearSuggestionFeatureConfigRequest', 2)