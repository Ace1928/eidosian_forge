from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsConversationModelsUndeployRequest(_messages.Message):
    """A DialogflowProjectsLocationsConversationModelsUndeployRequest object.

  Fields:
    googleCloudDialogflowV2UndeployConversationModelRequest: A
      GoogleCloudDialogflowV2UndeployConversationModelRequest resource to be
      passed as the request body.
    name: Required. The conversation model to undeploy. Format:
      `projects//conversationModels/`
  """
    googleCloudDialogflowV2UndeployConversationModelRequest = _messages.MessageField('GoogleCloudDialogflowV2UndeployConversationModelRequest', 1)
    name = _messages.StringField(2, required=True)