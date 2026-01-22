from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1Schema(_messages.Message):
    """Response for Schema call

  Fields:
    dimensions: List of schema fields grouped as dimensions.
    meta: Additional metadata associated with schema. This is a legacy field
      and usually consists of an empty array of strings.
    metrics: List of schema fields grouped as dimensions that can be used with
      an aggregate function such as `sum`, `avg`, `min`, and `max`.
  """
    dimensions = _messages.MessageField('GoogleCloudApigeeV1SchemaSchemaElement', 1, repeated=True)
    meta = _messages.StringField(2, repeated=True)
    metrics = _messages.MessageField('GoogleCloudApigeeV1SchemaSchemaElement', 3, repeated=True)