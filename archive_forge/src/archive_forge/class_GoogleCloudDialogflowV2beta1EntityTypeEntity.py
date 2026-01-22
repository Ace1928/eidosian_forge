from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1EntityTypeEntity(_messages.Message):
    """An **entity entry** for an associated entity type.

  Fields:
    synonyms: Required. A collection of value synonyms. For example, if the
      entity type is *vegetable*, and `value` is *scallions*, a synonym could
      be *green onions*. For `KIND_LIST` entity types: * This collection must
      contain exactly one synonym equal to `value`.
    value: Required. The primary value associated with this entity entry. For
      example, if the entity type is *vegetable*, the value could be
      *scallions*. For `KIND_MAP` entity types: * A reference value to be used
      in place of synonyms. For `KIND_LIST` entity types: * A string that can
      contain references to other entity types (with or without aliases).
  """
    synonyms = _messages.StringField(1, repeated=True)
    value = _messages.StringField(2)