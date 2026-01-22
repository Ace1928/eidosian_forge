from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1TaskInfrastructureSpecContainerImageRuntime(_messages.Message):
    """Container Image Runtime Configuration used with Batch execution.

  Messages:
    PropertiesValue: Optional. Override to common configuration of open source
      components installed on the Dataproc cluster. The properties to set on
      daemon config files. Property keys are specified in prefix:property
      format, for example core:hadoop.tmp.dir. For more information, see
      Cluster properties
      (https://cloud.google.com/dataproc/docs/concepts/cluster-properties).

  Fields:
    image: Optional. Container image to use.
    javaJars: Optional. A list of Java JARS to add to the classpath. Valid
      input includes Cloud Storage URIs to Jar binaries. For example,
      gs://bucket-name/my/path/to/file.jar
    properties: Optional. Override to common configuration of open source
      components installed on the Dataproc cluster. The properties to set on
      daemon config files. Property keys are specified in prefix:property
      format, for example core:hadoop.tmp.dir. For more information, see
      Cluster properties
      (https://cloud.google.com/dataproc/docs/concepts/cluster-properties).
    pythonPackages: Optional. A list of python packages to be installed. Valid
      formats include Cloud Storage URI to a PIP installable library. For
      example, gs://bucket-name/my/path/to/lib.tar.gz
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PropertiesValue(_messages.Message):
        """Optional. Override to common configuration of open source components
    installed on the Dataproc cluster. The properties to set on daemon config
    files. Property keys are specified in prefix:property format, for example
    core:hadoop.tmp.dir. For more information, see Cluster properties
    (https://cloud.google.com/dataproc/docs/concepts/cluster-properties).

    Messages:
      AdditionalProperty: An additional property for a PropertiesValue object.

    Fields:
      additionalProperties: Additional properties of type PropertiesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a PropertiesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    image = _messages.StringField(1)
    javaJars = _messages.StringField(2, repeated=True)
    properties = _messages.MessageField('PropertiesValue', 3)
    pythonPackages = _messages.StringField(4, repeated=True)