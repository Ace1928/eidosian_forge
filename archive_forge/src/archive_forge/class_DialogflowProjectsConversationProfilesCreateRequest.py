from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsConversationProfilesCreateRequest(_messages.Message):
    """A DialogflowProjectsConversationProfilesCreateRequest object.

  Fields:
    googleCloudDialogflowV2ConversationProfile: A
      GoogleCloudDialogflowV2ConversationProfile resource to be passed as the
      request body.
    parent: Required. The project to create a conversation profile for.
      Format: `projects//locations/`.
  """
    googleCloudDialogflowV2ConversationProfile = _messages.MessageField('GoogleCloudDialogflowV2ConversationProfile', 1)
    parent = _messages.StringField(2, required=True)