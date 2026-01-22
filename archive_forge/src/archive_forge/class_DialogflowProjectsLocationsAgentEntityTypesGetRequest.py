from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsAgentEntityTypesGetRequest(_messages.Message):
    """A DialogflowProjectsLocationsAgentEntityTypesGetRequest object.

  Fields:
    languageCode: Optional. The language used to access language-specific
      data. If not specified, the agent's default language is used. For more
      information, see [Multilingual intent and entity
      data](https://cloud.google.com/dialogflow/docs/agents-
      multilingual#intent-entity).
    name: Required. The name of the entity type. Format:
      `projects//agent/entityTypes/`.
  """
    languageCode = _messages.StringField(1)
    name = _messages.StringField(2, required=True)