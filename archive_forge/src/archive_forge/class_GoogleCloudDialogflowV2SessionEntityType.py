from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2SessionEntityType(_messages.Message):
    """A session represents a conversation between a Dialogflow agent and an
  end-user. You can create special entities, called session entities, during a
  session. Session entities can extend or replace custom entity types and only
  exist during the session that they were created for. All session data,
  including session entities, is stored by Dialogflow for 20 minutes. For more
  information, see the [session entity
  guide](https://cloud.google.com/dialogflow/docs/entities-session).

  Enums:
    EntityOverrideModeValueValuesEnum: Required. Indicates whether the
      additional data should override or supplement the custom entity type
      definition.

  Fields:
    entities: Required. The collection of entities associated with this
      session entity type.
    entityOverrideMode: Required. Indicates whether the additional data should
      override or supplement the custom entity type definition.
    name: Required. The unique identifier of this session entity type. Format:
      `projects//agent/sessions//entityTypes/`, or
      `projects//agent/environments//users//sessions//entityTypes/`. If
      `Environment ID` is not specified, we assume default 'draft'
      environment. If `User ID` is not specified, we assume default '-' user.
      `` must be the display name of an existing entity type in the same agent
      that will be overridden or supplemented.
  """

    class EntityOverrideModeValueValuesEnum(_messages.Enum):
        """Required. Indicates whether the additional data should override or
    supplement the custom entity type definition.

    Values:
      ENTITY_OVERRIDE_MODE_UNSPECIFIED: Not specified. This value should be
        never used.
      ENTITY_OVERRIDE_MODE_OVERRIDE: The collection of session entities
        overrides the collection of entities in the corresponding custom
        entity type.
      ENTITY_OVERRIDE_MODE_SUPPLEMENT: The collection of session entities
        extends the collection of entities in the corresponding custom entity
        type. Note: Even in this override mode calls to
        `ListSessionEntityTypes`, `GetSessionEntityType`,
        `CreateSessionEntityType` and `UpdateSessionEntityType` only return
        the additional entities added in this session entity type. If you want
        to get the supplemented list, please call EntityTypes.GetEntityType on
        the custom entity type and merge.
    """
        ENTITY_OVERRIDE_MODE_UNSPECIFIED = 0
        ENTITY_OVERRIDE_MODE_OVERRIDE = 1
        ENTITY_OVERRIDE_MODE_SUPPLEMENT = 2
    entities = _messages.MessageField('GoogleCloudDialogflowV2EntityTypeEntity', 1, repeated=True)
    entityOverrideMode = _messages.EnumField('EntityOverrideModeValueValuesEnum', 2)
    name = _messages.StringField(3)