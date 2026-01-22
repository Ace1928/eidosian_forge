from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsConversationModelsEvaluationsCreateRequest(_messages.Message):
    """A DialogflowProjectsLocationsConversationModelsEvaluationsCreateRequest
  object.

  Fields:
    googleCloudDialogflowV2CreateConversationModelEvaluationRequest: A
      GoogleCloudDialogflowV2CreateConversationModelEvaluationRequest resource
      to be passed as the request body.
    parent: Required. The conversation model resource name. Format:
      `projects//locations//conversationModels/`
  """
    googleCloudDialogflowV2CreateConversationModelEvaluationRequest = _messages.MessageField('GoogleCloudDialogflowV2CreateConversationModelEvaluationRequest', 1)
    parent = _messages.StringField(2, required=True)