from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2BatchDeleteEntitiesRequest(_messages.Message):
    """The request message for EntityTypes.BatchDeleteEntities.

  Fields:
    entityValues: Required. The reference `values` of the entities to delete.
      Note that these are not fully-qualified names, i.e. they don't start
      with `projects/`.
    languageCode: Optional. The language used to access language-specific
      data. If not specified, the agent's default language is used. For more
      information, see [Multilingual intent and entity
      data](https://cloud.google.com/dialogflow/docs/agents-
      multilingual#intent-entity).
  """
    entityValues = _messages.StringField(1, repeated=True)
    languageCode = _messages.StringField(2)