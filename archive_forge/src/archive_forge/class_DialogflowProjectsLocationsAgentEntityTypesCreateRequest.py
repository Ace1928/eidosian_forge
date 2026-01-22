from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsAgentEntityTypesCreateRequest(_messages.Message):
    """A DialogflowProjectsLocationsAgentEntityTypesCreateRequest object.

  Fields:
    googleCloudDialogflowV2EntityType: A GoogleCloudDialogflowV2EntityType
      resource to be passed as the request body.
    languageCode: Optional. The language used to access language-specific
      data. If not specified, the agent's default language is used. For more
      information, see [Multilingual intent and entity
      data](https://cloud.google.com/dialogflow/docs/agents-
      multilingual#intent-entity).
    parent: Required. The agent to create a entity type for. Format:
      `projects//agent`.
  """
    googleCloudDialogflowV2EntityType = _messages.MessageField('GoogleCloudDialogflowV2EntityType', 1)
    languageCode = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)