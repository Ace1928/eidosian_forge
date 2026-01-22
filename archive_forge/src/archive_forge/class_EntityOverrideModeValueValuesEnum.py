from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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