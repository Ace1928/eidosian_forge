from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GroupResult(_messages.Message):
    """Result containing the properties and count of a groupBy request.

  Messages:
    PropertiesValue: Properties matching the groupBy fields in the request.

  Fields:
    count: Total count of resources for the given properties.
    properties: Properties matching the groupBy fields in the request.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PropertiesValue(_messages.Message):
        """Properties matching the groupBy fields in the request.

    Messages:
      AdditionalProperty: An additional property for a PropertiesValue object.

    Fields:
      additionalProperties: Additional properties of type PropertiesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a PropertiesValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    count = _messages.IntegerField(1)
    properties = _messages.MessageField('PropertiesValue', 2)