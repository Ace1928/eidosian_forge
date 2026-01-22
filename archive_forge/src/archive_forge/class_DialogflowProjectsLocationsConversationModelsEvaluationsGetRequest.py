from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsConversationModelsEvaluationsGetRequest(_messages.Message):
    """A DialogflowProjectsLocationsConversationModelsEvaluationsGetRequest
  object.

  Fields:
    name: Required. The conversation model evaluation resource name. Format:
      `projects//conversationModels//evaluations/`
  """
    name = _messages.StringField(1, required=True)