from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TaskResult(_messages.Message):
    """TaskResult is used to describe the results of a task.

  Enums:
    TypeValueValuesEnum: The type of data that the result holds.

  Messages:
    PropertiesValue: When type is OBJECT, this map holds the names of fields
      inside that object along with the type of data each field holds.

  Fields:
    description: Description of the result.
    name: Name of the result.
    properties: When type is OBJECT, this map holds the names of fields inside
      that object along with the type of data each field holds.
    type: The type of data that the result holds.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """The type of data that the result holds.

    Values:
      TYPE_UNSPECIFIED: Default enum type; should not be used.
      STRING: Default
      ARRAY: Array type
      OBJECT: Object type
    """
        TYPE_UNSPECIFIED = 0
        STRING = 1
        ARRAY = 2
        OBJECT = 3

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PropertiesValue(_messages.Message):
        """When type is OBJECT, this map holds the names of fields inside that
    object along with the type of data each field holds.

    Messages:
      AdditionalProperty: An additional property for a PropertiesValue object.

    Fields:
      additionalProperties: Additional properties of type PropertiesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a PropertiesValue object.

      Fields:
        key: Name of the additional property.
        value: A PropertySpec attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('PropertySpec', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    description = _messages.StringField(1)
    name = _messages.StringField(2)
    properties = _messages.MessageField('PropertiesValue', 3)
    type = _messages.EnumField('TypeValueValuesEnum', 4)