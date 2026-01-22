from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1EntityType(_messages.Message):
    """Each intent parameter has a type, called the entity type, which dictates
  exactly how data from an end-user expression is extracted. Dialogflow
  provides predefined system entities that can match many common types of
  data. For example, there are system entities for matching dates, times,
  colors, email addresses, and so on. You can also create your own custom
  entities for matching custom data. For example, you could define a vegetable
  entity that can match the types of vegetables available for purchase with a
  grocery store agent. For more information, see the [Entity
  guide](https://cloud.google.com/dialogflow/docs/entities-overview).

  Enums:
    AutoExpansionModeValueValuesEnum: Optional. Indicates whether the entity
      type can be automatically expanded.
    KindValueValuesEnum: Required. Indicates the kind of entity type.

  Fields:
    autoExpansionMode: Optional. Indicates whether the entity type can be
      automatically expanded.
    displayName: Required. The name of the entity type.
    enableFuzzyExtraction: Optional. Enables fuzzy entity extraction during
      classification.
    entities: Optional. The collection of entity entries associated with the
      entity type.
    kind: Required. Indicates the kind of entity type.
    name: The unique identifier of the entity type. Required for
      EntityTypes.UpdateEntityType and EntityTypes.BatchUpdateEntityTypes
      methods. Supported formats: - `projects//agent/entityTypes/` -
      `projects//locations//agent/entityTypes/`
  """

    class AutoExpansionModeValueValuesEnum(_messages.Enum):
        """Optional. Indicates whether the entity type can be automatically
    expanded.

    Values:
      AUTO_EXPANSION_MODE_UNSPECIFIED: Auto expansion disabled for the entity.
      AUTO_EXPANSION_MODE_DEFAULT: Allows an agent to recognize values that
        have not been explicitly listed in the entity.
    """
        AUTO_EXPANSION_MODE_UNSPECIFIED = 0
        AUTO_EXPANSION_MODE_DEFAULT = 1

    class KindValueValuesEnum(_messages.Enum):
        """Required. Indicates the kind of entity type.

    Values:
      KIND_UNSPECIFIED: Not specified. This value should be never used.
      KIND_MAP: Map entity types allow mapping of a group of synonyms to a
        reference value.
      KIND_LIST: List entity types contain a set of entries that do not map to
        reference values. However, list entity types can contain references to
        other entity types (with or without aliases).
      KIND_REGEXP: Regexp entity types allow to specify regular expressions in
        entries values.
    """
        KIND_UNSPECIFIED = 0
        KIND_MAP = 1
        KIND_LIST = 2
        KIND_REGEXP = 3
    autoExpansionMode = _messages.EnumField('AutoExpansionModeValueValuesEnum', 1)
    displayName = _messages.StringField(2)
    enableFuzzyExtraction = _messages.BooleanField(3)
    entities = _messages.MessageField('GoogleCloudDialogflowV2beta1EntityTypeEntity', 4, repeated=True)
    kind = _messages.EnumField('KindValueValuesEnum', 5)
    name = _messages.StringField(6)