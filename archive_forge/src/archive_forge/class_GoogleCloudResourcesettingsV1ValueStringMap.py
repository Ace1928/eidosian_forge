from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GoogleCloudResourcesettingsV1ValueStringMap(_messages.Message):
    """A string->string map value that can hold a map of string keys to string
  values. The maximum length of each string is 200 characters and there can be
  a maximum of 50 key-value pairs in the map.

  Messages:
    MappingsValue: The key-value pairs in the map

  Fields:
    mappings: The key-value pairs in the map
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MappingsValue(_messages.Message):
        """The key-value pairs in the map

    Messages:
      AdditionalProperty: An additional property for a MappingsValue object.

    Fields:
      additionalProperties: Additional properties of type MappingsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MappingsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    mappings = _messages.MessageField('MappingsValue', 1)