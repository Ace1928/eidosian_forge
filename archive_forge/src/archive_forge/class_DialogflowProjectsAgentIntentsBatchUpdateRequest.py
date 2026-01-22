from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsAgentIntentsBatchUpdateRequest(_messages.Message):
    """A DialogflowProjectsAgentIntentsBatchUpdateRequest object.

  Fields:
    googleCloudDialogflowV2BatchUpdateIntentsRequest: A
      GoogleCloudDialogflowV2BatchUpdateIntentsRequest resource to be passed
      as the request body.
    parent: Required. The name of the agent to update or create intents in.
      Format: `projects//agent`.
  """
    googleCloudDialogflowV2BatchUpdateIntentsRequest = _messages.MessageField('GoogleCloudDialogflowV2BatchUpdateIntentsRequest', 1)
    parent = _messages.StringField(2, required=True)