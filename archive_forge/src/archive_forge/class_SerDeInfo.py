from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SerDeInfo(_messages.Message):
    """Serializer and deserializer information.

  Messages:
    ParametersValue: Optional. Key-value pairs that define the initialization
      parameters for the serialization library. Maximum size 10 Kib.

  Fields:
    name: Optional. Name of the SerDe. The maximum length is 256 characters.
    parameters: Optional. Key-value pairs that define the initialization
      parameters for the serialization library. Maximum size 10 Kib.
    serializationLibrary: Required. Specifies a fully-qualified class name of
      the serialization library that is responsible for the translation of
      data between table representation and the underlying low-level input and
      output format structures. The maximum length is 256 characters.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ParametersValue(_messages.Message):
        """Optional. Key-value pairs that define the initialization parameters
    for the serialization library. Maximum size 10 Kib.

    Messages:
      AdditionalProperty: An additional property for a ParametersValue object.

    Fields:
      additionalProperties: Additional properties of type ParametersValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ParametersValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    name = _messages.StringField(1)
    parameters = _messages.MessageField('ParametersValue', 2)
    serializationLibrary = _messages.StringField(3)