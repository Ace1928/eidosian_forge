from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsAgentEntityTypesBatchUpdateRequest(_messages.Message):
    """A DialogflowProjectsAgentEntityTypesBatchUpdateRequest object.

  Fields:
    googleCloudDialogflowV2BatchUpdateEntityTypesRequest: A
      GoogleCloudDialogflowV2BatchUpdateEntityTypesRequest resource to be
      passed as the request body.
    parent: Required. The name of the agent to update or create entity types
      in. Format: `projects//agent`.
  """
    googleCloudDialogflowV2BatchUpdateEntityTypesRequest = _messages.MessageField('GoogleCloudDialogflowV2BatchUpdateEntityTypesRequest', 1)
    parent = _messages.StringField(2, required=True)