from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class TriggerFilter(_messages.Message):
    """Filters events based on exact matches on the attributes.

  Messages:
    AttributesValue: Attributes filters events by exact match on event context
      attributes. Each key in the map is compared with the equivalent key in
      the event context. An event passes the filter if all values are equal to
      the specified values. Nested context attributes are not supported as
      keys. Only string values are supported.

  Fields:
    attributes: Attributes filters events by exact match on event context
      attributes. Each key in the map is compared with the equivalent key in
      the event context. An event passes the filter if all values are equal to
      the specified values. Nested context attributes are not supported as
      keys. Only string values are supported.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AttributesValue(_messages.Message):
        """Attributes filters events by exact match on event context attributes.
    Each key in the map is compared with the equivalent key in the event
    context. An event passes the filter if all values are equal to the
    specified values. Nested context attributes are not supported as keys.
    Only string values are supported.

    Messages:
      AdditionalProperty: An additional property for a AttributesValue object.

    Fields:
      additionalProperties: Additional properties of type AttributesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AttributesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    attributes = _messages.MessageField('AttributesValue', 1)