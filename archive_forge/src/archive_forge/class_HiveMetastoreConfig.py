from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HiveMetastoreConfig(_messages.Message):
    """Specifies configuration information specific to running Hive metastore
  software as the metastore service.

  Enums:
    EndpointProtocolValueValuesEnum: The protocol to use for the metastore
      service endpoint. If unspecified, defaults to THRIFT.

  Messages:
    AuxiliaryVersionsValue: A mapping of Hive metastore version to the
      auxiliary version configuration. When specified, a secondary Hive
      metastore service is created along with the primary service. All
      auxiliary versions must be less than the service's primary version. The
      key is the auxiliary service name and it must match the regular
      expression a-z?. This means that the first character must be a lowercase
      letter, and all the following characters must be hyphens, lowercase
      letters, or digits, except the last character, which cannot be a hyphen.
    ConfigOverridesValue: A mapping of Hive metastore configuration key-value
      pairs to apply to the Hive metastore (configured in hive-site.xml). The
      mappings override system defaults (some keys cannot be overridden).
      These overrides are also applied to auxiliary versions and can be
      further customized in the auxiliary version's AuxiliaryVersionConfig.

  Fields:
    auxiliaryVersions: A mapping of Hive metastore version to the auxiliary
      version configuration. When specified, a secondary Hive metastore
      service is created along with the primary service. All auxiliary
      versions must be less than the service's primary version. The key is the
      auxiliary service name and it must match the regular expression a-z?.
      This means that the first character must be a lowercase letter, and all
      the following characters must be hyphens, lowercase letters, or digits,
      except the last character, which cannot be a hyphen.
    configOverrides: A mapping of Hive metastore configuration key-value pairs
      to apply to the Hive metastore (configured in hive-site.xml). The
      mappings override system defaults (some keys cannot be overridden).
      These overrides are also applied to auxiliary versions and can be
      further customized in the auxiliary version's AuxiliaryVersionConfig.
    endpointProtocol: The protocol to use for the metastore service endpoint.
      If unspecified, defaults to THRIFT.
    kerberosConfig: Information used to configure the Hive metastore service
      as a service principal in a Kerberos realm. To disable Kerberos, use the
      UpdateService method and specify this field's path
      (hive_metastore_config.kerberos_config) in the request's update_mask
      while omitting this field from the request's service.
    version: Immutable. The Hive metastore schema version.
  """

    class EndpointProtocolValueValuesEnum(_messages.Enum):
        """The protocol to use for the metastore service endpoint. If
    unspecified, defaults to THRIFT.

    Values:
      ENDPOINT_PROTOCOL_UNSPECIFIED: The protocol is not set.
      THRIFT: Use the legacy Apache Thrift protocol for the metastore service
        endpoint.
      GRPC: Use the modernized gRPC protocol for the metastore service
        endpoint.
    """
        ENDPOINT_PROTOCOL_UNSPECIFIED = 0
        THRIFT = 1
        GRPC = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AuxiliaryVersionsValue(_messages.Message):
        """A mapping of Hive metastore version to the auxiliary version
    configuration. When specified, a secondary Hive metastore service is
    created along with the primary service. All auxiliary versions must be
    less than the service's primary version. The key is the auxiliary service
    name and it must match the regular expression a-z?. This means that the
    first character must be a lowercase letter, and all the following
    characters must be hyphens, lowercase letters, or digits, except the last
    character, which cannot be a hyphen.

    Messages:
      AdditionalProperty: An additional property for a AuxiliaryVersionsValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        AuxiliaryVersionsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AuxiliaryVersionsValue object.

      Fields:
        key: Name of the additional property.
        value: A AuxiliaryVersionConfig attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('AuxiliaryVersionConfig', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ConfigOverridesValue(_messages.Message):
        """A mapping of Hive metastore configuration key-value pairs to apply to
    the Hive metastore (configured in hive-site.xml). The mappings override
    system defaults (some keys cannot be overridden). These overrides are also
    applied to auxiliary versions and can be further customized in the
    auxiliary version's AuxiliaryVersionConfig.

    Messages:
      AdditionalProperty: An additional property for a ConfigOverridesValue
        object.

    Fields:
      additionalProperties: Additional properties of type ConfigOverridesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ConfigOverridesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    auxiliaryVersions = _messages.MessageField('AuxiliaryVersionsValue', 1)
    configOverrides = _messages.MessageField('ConfigOverridesValue', 2)
    endpointProtocol = _messages.EnumField('EndpointProtocolValueValuesEnum', 3)
    kerberosConfig = _messages.MessageField('KerberosConfig', 4)
    version = _messages.StringField(5)