from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsAgentEntityTypesPatchRequest(_messages.Message):
    """A DialogflowProjectsLocationsAgentEntityTypesPatchRequest object.

  Fields:
    googleCloudDialogflowV2EntityType: A GoogleCloudDialogflowV2EntityType
      resource to be passed as the request body.
    languageCode: Optional. The language used to access language-specific
      data. If not specified, the agent's default language is used. For more
      information, see [Multilingual intent and entity
      data](https://cloud.google.com/dialogflow/docs/agents-
      multilingual#intent-entity).
    name: The unique identifier of the entity type. Required for
      EntityTypes.UpdateEntityType and EntityTypes.BatchUpdateEntityTypes
      methods. Format: `projects//agent/entityTypes/`.
    updateMask: Optional. The mask to control which fields get updated.
  """
    googleCloudDialogflowV2EntityType = _messages.MessageField('GoogleCloudDialogflowV2EntityType', 1)
    languageCode = _messages.StringField(2)
    name = _messages.StringField(3, required=True)
    updateMask = _messages.StringField(4)