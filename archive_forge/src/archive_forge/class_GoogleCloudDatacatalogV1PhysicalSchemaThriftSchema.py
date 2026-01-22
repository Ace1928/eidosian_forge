from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1PhysicalSchemaThriftSchema(_messages.Message):
    """Schema in Thrift format.

  Fields:
    text: Thrift IDL source of the schema.
  """
    text = _messages.StringField(1)