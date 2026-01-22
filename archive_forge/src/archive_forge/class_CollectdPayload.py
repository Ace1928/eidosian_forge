from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CollectdPayload(_messages.Message):
    """A collection of data points sent from a collectd-based plugin. See the
  collectd documentation for more information.

  Messages:
    MetadataValue: The measurement metadata. Example: "process_id" -> 12345

  Fields:
    endTime: The end time of the interval.
    metadata: The measurement metadata. Example: "process_id" -> 12345
    plugin: The name of the plugin. Example: "disk".
    pluginInstance: The instance name of the plugin Example: "hdcl".
    startTime: The start time of the interval.
    type: The measurement type. Example: "memory".
    typeInstance: The measurement type instance. Example: "used".
    values: The measured values during this time interval. Each value must
      have a different data_source_name.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MetadataValue(_messages.Message):
        """The measurement metadata. Example: "process_id" -> 12345

    Messages:
      AdditionalProperty: An additional property for a MetadataValue object.

    Fields:
      additionalProperties: Additional properties of type MetadataValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MetadataValue object.

      Fields:
        key: Name of the additional property.
        value: A TypedValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('TypedValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    endTime = _messages.StringField(1)
    metadata = _messages.MessageField('MetadataValue', 2)
    plugin = _messages.StringField(3)
    pluginInstance = _messages.StringField(4)
    startTime = _messages.StringField(5)
    type = _messages.StringField(6)
    typeInstance = _messages.StringField(7)
    values = _messages.MessageField('CollectdValue', 8, repeated=True)