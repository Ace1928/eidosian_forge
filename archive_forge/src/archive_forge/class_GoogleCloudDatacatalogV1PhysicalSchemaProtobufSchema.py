from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1PhysicalSchemaProtobufSchema(_messages.Message):
    """Schema in protocol buffer format.

  Fields:
    text: Protocol buffer source of the schema.
  """
    text = _messages.StringField(1)