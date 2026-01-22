from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1DocumentSchemaEntityType(_messages.Message):
    """EntityType is the wrapper of a label of the corresponding model with
  detailed attributes and limitations for entity-based processors. Multiple
  types can also compose a dependency tree to represent nested types.

  Fields:
    baseTypes: The entity type that this type is derived from. For now, one
      and only one should be set.
    displayName: User defined name for the type.
    enumValues: If specified, lists all the possible values for this entity.
      This should not be more than a handful of values. If the number of
      values is >10 or could change frequently use the
      `EntityType.value_ontology` field and specify a list of all possible
      values in a value ontology file.
    name: Name of the type. It must be unique within the schema file and
      cannot be a "Common Type". The following naming conventions are used: -
      Use `snake_casing`. - Name matching is case-sensitive. - Maximum 64
      characters. - Must start with a letter. - Allowed characters: ASCII
      letters `[a-z0-9_-]`. (For backward compatibility internal
      infrastructure and tooling can handle any ascii character.) - The `/` is
      sometimes used to denote a property of a type. For example
      `line_item/amount`. This convention is deprecated, but will still be
      honored for backward compatibility.
    properties: Description the nested structure, or composition of an entity.
  """
    baseTypes = _messages.StringField(1, repeated=True)
    displayName = _messages.StringField(2)
    enumValues = _messages.MessageField('GoogleCloudDocumentaiV1DocumentSchemaEntityTypeEnumValues', 3)
    name = _messages.StringField(4)
    properties = _messages.MessageField('GoogleCloudDocumentaiV1DocumentSchemaEntityTypeProperty', 5, repeated=True)