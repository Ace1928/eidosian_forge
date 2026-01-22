from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1FieldTypeEnumTypeEnumValue(_messages.Message):
    """A GoogleCloudDatacatalogV1FieldTypeEnumTypeEnumValue object.

  Fields:
    displayName: Required. The display name of the enum value. Must not be an
      empty string. The name must contain only Unicode letters, numbers (0-9),
      underscores (_), dashes (-), spaces ( ), and can't start or end with
      spaces. The maximum length is 200 characters.
  """
    displayName = _messages.StringField(1)