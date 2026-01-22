from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1TargetServerConfig(_messages.Message):
    """A GoogleCloudApigeeV1TargetServerConfig object.

  Enums:
    ProtocolValueValuesEnum: The protocol used by this target server.

  Fields:
    enabled: Whether the target server is enabled. An empty/omitted value for
      this field should be interpreted as true.
    host: Host name of the target server.
    name: Target server revision name in the following format: `organizations/
      {org}/environments/{env}/targetservers/{targetserver}/revisions/{rev}`
    port: Port number for the target server.
    protocol: The protocol used by this target server.
    tlsInfo: TLS settings for the target server.
  """

    class ProtocolValueValuesEnum(_messages.Enum):
        """The protocol used by this target server.

    Values:
      PROTOCOL_UNSPECIFIED: UNSPECIFIED defaults to HTTP for backwards
        compatibility.
      HTTP: The TargetServer uses HTTP.
      HTTP2: The TargetSever uses HTTP2.
      GRPC_TARGET: The TargetServer uses GRPC.
      GRPC: GRPC TargetServer to be used in ExternalCallout Policy. Prefer to
        use EXTERNAL_CALLOUT instead. TODO(b/266125112) deprecate once
        EXTERNAL _CALLOUT generally available.
      EXTERNAL_CALLOUT: The TargetServer is to be used in the ExternalCallout
        Policy
    """
        PROTOCOL_UNSPECIFIED = 0
        HTTP = 1
        HTTP2 = 2
        GRPC_TARGET = 3
        GRPC = 4
        EXTERNAL_CALLOUT = 5
    enabled = _messages.BooleanField(1)
    host = _messages.StringField(2)
    name = _messages.StringField(3)
    port = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    protocol = _messages.EnumField('ProtocolValueValuesEnum', 5)
    tlsInfo = _messages.MessageField('GoogleCloudApigeeV1TlsInfoConfig', 6)