from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class JSONSchemaPropsOrStringArray(_messages.Message):
    """JSONSchemaPropsOrStringArray represents a JSONSchemaProps or a string
  array.

  Fields:
    property: A string attribute.
    schema: A JSONSchemaProps attribute.
  """
    property = _messages.StringField(1, repeated=True)
    schema = _messages.MessageField('JSONSchemaProps', 2)