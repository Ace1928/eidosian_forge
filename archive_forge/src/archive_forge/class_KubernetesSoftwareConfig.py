from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KubernetesSoftwareConfig(_messages.Message):
    """The software configuration for this Dataproc cluster running on
  Kubernetes.

  Messages:
    ComponentVersionValue: The components that should be installed in this
      Dataproc cluster. The key must be a string from the KubernetesComponent
      enumeration. The value is the version of the software to be installed.
      At least one entry must be specified.
    PropertiesValue: The properties to set on daemon config files.Property
      keys are specified in prefix:property format, for example
      spark:spark.kubernetes.container.image. The following are supported
      prefixes and their mappings: spark: spark-defaults.confFor more
      information, see Cluster properties
      (https://cloud.google.com/dataproc/docs/concepts/cluster-properties).

  Fields:
    componentVersion: The components that should be installed in this Dataproc
      cluster. The key must be a string from the KubernetesComponent
      enumeration. The value is the version of the software to be installed.
      At least one entry must be specified.
    properties: The properties to set on daemon config files.Property keys are
      specified in prefix:property format, for example
      spark:spark.kubernetes.container.image. The following are supported
      prefixes and their mappings: spark: spark-defaults.confFor more
      information, see Cluster properties
      (https://cloud.google.com/dataproc/docs/concepts/cluster-properties).
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ComponentVersionValue(_messages.Message):
        """The components that should be installed in this Dataproc cluster. The
    key must be a string from the KubernetesComponent enumeration. The value
    is the version of the software to be installed. At least one entry must be
    specified.

    Messages:
      AdditionalProperty: An additional property for a ComponentVersionValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        ComponentVersionValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ComponentVersionValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PropertiesValue(_messages.Message):
        """The properties to set on daemon config files.Property keys are
    specified in prefix:property format, for example
    spark:spark.kubernetes.container.image. The following are supported
    prefixes and their mappings: spark: spark-defaults.confFor more
    information, see Cluster properties
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
    componentVersion = _messages.MessageField('ComponentVersionValue', 1)
    properties = _messages.MessageField('PropertiesValue', 2)