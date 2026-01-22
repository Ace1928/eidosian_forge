from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
@encoding.MapUnrecognizedFields('additionalProperties')
class PatternPropertiesValue(_messages.Message):
    """A PatternPropertiesValue object.

    Messages:
      AdditionalProperty: An additional property for a PatternPropertiesValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        PatternPropertiesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a PatternPropertiesValue object.

      Fields:
        key: Name of the additional property.
        value: A JSONSchemaProps attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('JSONSchemaProps', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)