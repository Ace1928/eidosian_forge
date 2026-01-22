from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryParameterValue(_messages.Message):
    """The value of a query parameter.

  Messages:
    StructValuesValue: The struct field values.

  Fields:
    arrayValues: Optional. The array values, if this is an array type.
    rangeValue: Optional. The range value, if this is a range type.
    structValues: The struct field values.
    value: Optional. The value of this value, if a simple scalar type.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class StructValuesValue(_messages.Message):
        """The struct field values.

    Messages:
      AdditionalProperty: An additional property for a StructValuesValue
        object.

    Fields:
      additionalProperties: Additional properties of type StructValuesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a StructValuesValue object.

      Fields:
        key: Name of the additional property.
        value: A QueryParameterValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('QueryParameterValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    arrayValues = _messages.MessageField('QueryParameterValue', 1, repeated=True)
    rangeValue = _messages.MessageField('RangeValue', 2)
    structValues = _messages.MessageField('StructValuesValue', 3)
    value = _messages.StringField(4)