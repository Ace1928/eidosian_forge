from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SparkHistoryServerConfig(_messages.Message):
    """Spark History Server Config.

  Messages:
    ConfigurationsValue: Optional. Mapping of configurations.

  Fields:
    configurations: Optional. Mapping of configurations.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ConfigurationsValue(_messages.Message):
        """Optional. Mapping of configurations.

    Messages:
      AdditionalProperty: An additional property for a ConfigurationsValue
        object.

    Fields:
      additionalProperties: Additional properties of type ConfigurationsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ConfigurationsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    configurations = _messages.MessageField('ConfigurationsValue', 1)