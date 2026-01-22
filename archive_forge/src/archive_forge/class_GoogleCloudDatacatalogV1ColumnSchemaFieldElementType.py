from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1ColumnSchemaFieldElementType(_messages.Message):
    """Represents the type of a field element.

  Fields:
    type: Required. The type of a field element. See ColumnSchema.type.
  """
    type = _messages.StringField(1)