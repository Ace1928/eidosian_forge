from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GatewayConfig(_messages.Message):
    """Gateway-related configuration and state.

  Enums:
    GatewayAuthMethodValueValuesEnum: Indicates how to authorize and/or
      authenticate devices to access the gateway.
    GatewayTypeValueValuesEnum: Indicates whether the device is a gateway.

  Fields:
    gatewayAuthMethod: Indicates how to authorize and/or authenticate devices
      to access the gateway.
    gatewayType: Indicates whether the device is a gateway.
    lastAccessedGatewayId: [Output only] The ID of the gateway the device
      accessed most recently.
    lastAccessedGatewayTime: [Output only] The most recent time at which the
      device accessed the gateway specified in `last_accessed_gateway`.
  """

    class GatewayAuthMethodValueValuesEnum(_messages.Enum):
        """Indicates how to authorize and/or authenticate devices to access the
    gateway.

    Values:
      GATEWAY_AUTH_METHOD_UNSPECIFIED: No authentication/authorization method
        specified. No devices are allowed to access the gateway.
      ASSOCIATION_ONLY: The device is authenticated through the gateway
        association only. Device credentials are ignored even if provided.
      DEVICE_AUTH_TOKEN_ONLY: The device is authenticated through its own
        credentials. Gateway association is not checked.
      ASSOCIATION_AND_DEVICE_AUTH_TOKEN: The device is authenticated through
        both device credentials and gateway association. The device must be
        bound to the gateway and must provide its own credentials.
    """
        GATEWAY_AUTH_METHOD_UNSPECIFIED = 0
        ASSOCIATION_ONLY = 1
        DEVICE_AUTH_TOKEN_ONLY = 2
        ASSOCIATION_AND_DEVICE_AUTH_TOKEN = 3

    class GatewayTypeValueValuesEnum(_messages.Enum):
        """Indicates whether the device is a gateway.

    Values:
      GATEWAY_TYPE_UNSPECIFIED: If unspecified, the device is considered a
        non-gateway device.
      GATEWAY: The device is a gateway.
      NON_GATEWAY: The device is not a gateway.
    """
        GATEWAY_TYPE_UNSPECIFIED = 0
        GATEWAY = 1
        NON_GATEWAY = 2
    gatewayAuthMethod = _messages.EnumField('GatewayAuthMethodValueValuesEnum', 1)
    gatewayType = _messages.EnumField('GatewayTypeValueValuesEnum', 2)
    lastAccessedGatewayId = _messages.StringField(3)
    lastAccessedGatewayTime = _messages.StringField(4)