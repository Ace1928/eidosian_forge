from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1RenameTagTemplateFieldEnumValueRequest(_messages.Message):
    """Request message for RenameTagTemplateFieldEnumValue.

  Fields:
    newEnumValueDisplayName: Required. The new display name of the enum value.
      For example, `my_new_enum_value`.
  """
    newEnumValueDisplayName = _messages.StringField(1)