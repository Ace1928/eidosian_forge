from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsConversationProfilesSetSuggestionFeatureConfigRequest(_messages.Message):
    """A DialogflowProjectsLocationsConversationProfilesSetSuggestionFeatureCon
  figRequest object.

  Fields:
    conversationProfile: Required. The Conversation Profile to add or update
      the suggestion feature config. Format:
      `projects//locations//conversationProfiles/`.
    googleCloudDialogflowV2SetSuggestionFeatureConfigRequest: A
      GoogleCloudDialogflowV2SetSuggestionFeatureConfigRequest resource to be
      passed as the request body.
  """
    conversationProfile = _messages.StringField(1, required=True)
    googleCloudDialogflowV2SetSuggestionFeatureConfigRequest = _messages.MessageField('GoogleCloudDialogflowV2SetSuggestionFeatureConfigRequest', 2)