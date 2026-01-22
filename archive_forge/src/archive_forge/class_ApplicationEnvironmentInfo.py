from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApplicationEnvironmentInfo(_messages.Message):
    """Details about the Environment that the application is running in.

  Messages:
    ClasspathEntriesValue: A ClasspathEntriesValue object.
    HadoopPropertiesValue: A HadoopPropertiesValue object.
    MetricsPropertiesValue: A MetricsPropertiesValue object.
    SparkPropertiesValue: A SparkPropertiesValue object.
    SystemPropertiesValue: A SystemPropertiesValue object.

  Fields:
    classpathEntries: A ClasspathEntriesValue attribute.
    hadoopProperties: A HadoopPropertiesValue attribute.
    metricsProperties: A MetricsPropertiesValue attribute.
    resourceProfiles: A ResourceProfileInfo attribute.
    runtime: A SparkRuntimeInfo attribute.
    sparkProperties: A SparkPropertiesValue attribute.
    systemProperties: A SystemPropertiesValue attribute.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ClasspathEntriesValue(_messages.Message):
        """A ClasspathEntriesValue object.

    Messages:
      AdditionalProperty: An additional property for a ClasspathEntriesValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        ClasspathEntriesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ClasspathEntriesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class HadoopPropertiesValue(_messages.Message):
        """A HadoopPropertiesValue object.

    Messages:
      AdditionalProperty: An additional property for a HadoopPropertiesValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        HadoopPropertiesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a HadoopPropertiesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MetricsPropertiesValue(_messages.Message):
        """A MetricsPropertiesValue object.

    Messages:
      AdditionalProperty: An additional property for a MetricsPropertiesValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        MetricsPropertiesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MetricsPropertiesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class SparkPropertiesValue(_messages.Message):
        """A SparkPropertiesValue object.

    Messages:
      AdditionalProperty: An additional property for a SparkPropertiesValue
        object.

    Fields:
      additionalProperties: Additional properties of type SparkPropertiesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a SparkPropertiesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class SystemPropertiesValue(_messages.Message):
        """A SystemPropertiesValue object.

    Messages:
      AdditionalProperty: An additional property for a SystemPropertiesValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        SystemPropertiesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a SystemPropertiesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    classpathEntries = _messages.MessageField('ClasspathEntriesValue', 1)
    hadoopProperties = _messages.MessageField('HadoopPropertiesValue', 2)
    metricsProperties = _messages.MessageField('MetricsPropertiesValue', 3)
    resourceProfiles = _messages.MessageField('ResourceProfileInfo', 4, repeated=True)
    runtime = _messages.MessageField('SparkRuntimeInfo', 5)
    sparkProperties = _messages.MessageField('SparkPropertiesValue', 6)
    systemProperties = _messages.MessageField('SystemPropertiesValue', 7)