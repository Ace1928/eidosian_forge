from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsAgentEntityTypesEntitiesBatchCreateRequest(_messages.Message):
    """A DialogflowProjectsAgentEntityTypesEntitiesBatchCreateRequest object.

  Fields:
    googleCloudDialogflowV2BatchCreateEntitiesRequest: A
      GoogleCloudDialogflowV2BatchCreateEntitiesRequest resource to be passed
      as the request body.
    parent: Required. The name of the entity type to create entities in.
      Format: `projects//agent/entityTypes/`.
  """
    googleCloudDialogflowV2BatchCreateEntitiesRequest = _messages.MessageField('GoogleCloudDialogflowV2BatchCreateEntitiesRequest', 1)
    parent = _messages.StringField(2, required=True)