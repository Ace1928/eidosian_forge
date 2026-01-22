from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1FieldTypeEnumType(_messages.Message):
    """A GoogleCloudDatacatalogV1FieldTypeEnumType object.

  Fields:
    allowedValues: The set of allowed values for this enum. This set must not
      be empty and can include up to 100 allowed values. The display names of
      the values in this set must not be empty and must be case-insensitively
      unique within this set. The order of items in this set is preserved.
      This field can be used to create, remove, and reorder enum values. To
      rename enum values, use the `RenameTagTemplateFieldEnumValue` method.
  """
    allowedValues = _messages.MessageField('GoogleCloudDatacatalogV1FieldTypeEnumTypeEnumValue', 1, repeated=True)