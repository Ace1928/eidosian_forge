from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudEventOverrides(_messages.Message):
    """CloudEventOverrides defines arguments for a Source that control the
  output format of the CloudEvents produced by the Source.

  Messages:
    ExtensionsValue: Extensions specify what attribute are added or overridden
      on the outbound event. Each `Extensions` key-value pair are set on the
      event as an attribute extension independently.

  Fields:
    extensions: Extensions specify what attribute are added or overridden on
      the outbound event. Each `Extensions` key-value pair are set on the
      event as an attribute extension independently.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ExtensionsValue(_messages.Message):
        """Extensions specify what attribute are added or overridden on the
    outbound event. Each `Extensions` key-value pair are set on the event as
    an attribute extension independently.

    Messages:
      AdditionalProperty: An additional property for a ExtensionsValue object.

    Fields:
      additionalProperties: Additional properties of type ExtensionsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ExtensionsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    extensions = _messages.MessageField('ExtensionsValue', 1)