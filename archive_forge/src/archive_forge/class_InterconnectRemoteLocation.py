from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InterconnectRemoteLocation(_messages.Message):
    """Represents a Cross-Cloud Interconnect Remote Location resource. You can
  use this resource to find remote location details about an Interconnect
  attachment (VLAN).

  Enums:
    ContinentValueValuesEnum: [Output Only] Continent for this location, which
      can take one of the following values: - AFRICA - ASIA_PAC - EUROPE -
      NORTH_AMERICA - SOUTH_AMERICA
    LacpValueValuesEnum: [Output Only] Link Aggregation Control Protocol
      (LACP) constraints, which can take one of the following values:
      LACP_SUPPORTED, LACP_UNSUPPORTED
    StatusValueValuesEnum: [Output Only] The status of this
      InterconnectRemoteLocation, which can take one of the following values:
      - CLOSED: The InterconnectRemoteLocation is closed and is unavailable
      for provisioning new Cross-Cloud Interconnects. - AVAILABLE: The
      InterconnectRemoteLocation is available for provisioning new Cross-Cloud
      Interconnects.

  Fields:
    address: [Output Only] The postal address of the Point of Presence, each
      line in the address is separated by a newline character.
    attachmentConfigurationConstraints: [Output Only] Subset of fields from
      InterconnectAttachment's |configurationConstraints| field that apply to
      all attachments for this remote location.
    city: [Output Only] Metropolitan area designator that indicates which city
      an interconnect is located. For example: "Chicago, IL", "Amsterdam,
      Netherlands".
    constraints: [Output Only] Constraints on the parameters for creating
      Cross-Cloud Interconnect and associated InterconnectAttachments.
    continent: [Output Only] Continent for this location, which can take one
      of the following values: - AFRICA - ASIA_PAC - EUROPE - NORTH_AMERICA -
      SOUTH_AMERICA
    creationTimestamp: [Output Only] Creation timestamp in RFC3339 text
      format.
    description: [Output Only] An optional description of the resource.
    facilityProvider: [Output Only] The name of the provider for this facility
      (e.g., EQUINIX).
    facilityProviderFacilityId: [Output Only] A provider-assigned Identifier
      for this facility (e.g., Ashburn-DC1).
    id: [Output Only] The unique identifier for the resource. This identifier
      is defined by the server.
    kind: [Output Only] Type of the resource. Always
      compute#interconnectRemoteLocation for interconnect remote locations.
    lacp: [Output Only] Link Aggregation Control Protocol (LACP) constraints,
      which can take one of the following values: LACP_SUPPORTED,
      LACP_UNSUPPORTED
    maxLagSize100Gbps: [Output Only] The maximum number of 100 Gbps ports
      supported in a link aggregation group (LAG). When linkType is 100 Gbps,
      requestedLinkCount cannot exceed max_lag_size_100_gbps.
    maxLagSize10Gbps: [Output Only] The maximum number of 10 Gbps ports
      supported in a link aggregation group (LAG). When linkType is 10 Gbps,
      requestedLinkCount cannot exceed max_lag_size_10_gbps.
    name: [Output Only] Name of the resource.
    peeringdbFacilityId: [Output Only] The peeringdb identifier for this
      facility (corresponding with a netfac type in peeringdb).
    permittedConnections: [Output Only] Permitted connections.
    remoteService: [Output Only] Indicates the service provider present at the
      remote location. Example values: "Amazon Web Services", "Microsoft
      Azure".
    selfLink: [Output Only] Server-defined URL for the resource.
    status: [Output Only] The status of this InterconnectRemoteLocation, which
      can take one of the following values: - CLOSED: The
      InterconnectRemoteLocation is closed and is unavailable for provisioning
      new Cross-Cloud Interconnects. - AVAILABLE: The
      InterconnectRemoteLocation is available for provisioning new Cross-Cloud
      Interconnects.
  """

    class ContinentValueValuesEnum(_messages.Enum):
        """[Output Only] Continent for this location, which can take one of the
    following values: - AFRICA - ASIA_PAC - EUROPE - NORTH_AMERICA -
    SOUTH_AMERICA

    Values:
      AFRICA: <no description>
      ASIA_PAC: <no description>
      EUROPE: <no description>
      NORTH_AMERICA: <no description>
      SOUTH_AMERICA: <no description>
    """
        AFRICA = 0
        ASIA_PAC = 1
        EUROPE = 2
        NORTH_AMERICA = 3
        SOUTH_AMERICA = 4

    class LacpValueValuesEnum(_messages.Enum):
        """[Output Only] Link Aggregation Control Protocol (LACP) constraints,
    which can take one of the following values: LACP_SUPPORTED,
    LACP_UNSUPPORTED

    Values:
      LACP_SUPPORTED: LACP_SUPPORTED: LACP is supported, and enabled by
        default on the Cross-Cloud Interconnect.
      LACP_UNSUPPORTED: LACP_UNSUPPORTED: LACP is not supported and is not be
        enabled on this port. GetDiagnostics shows bundleAggregationType as
        "static". GCP does not support LAGs without LACP, so
        requestedLinkCount must be 1.
    """
        LACP_SUPPORTED = 0
        LACP_UNSUPPORTED = 1

    class StatusValueValuesEnum(_messages.Enum):
        """[Output Only] The status of this InterconnectRemoteLocation, which can
    take one of the following values: - CLOSED: The InterconnectRemoteLocation
    is closed and is unavailable for provisioning new Cross-Cloud
    Interconnects. - AVAILABLE: The InterconnectRemoteLocation is available
    for provisioning new Cross-Cloud Interconnects.

    Values:
      AVAILABLE: The InterconnectRemoteLocation is available for provisioning
        new Cross-Cloud Interconnects.
      CLOSED: The InterconnectRemoteLocation is closed for provisioning new
        Cross-Cloud Interconnects.
    """
        AVAILABLE = 0
        CLOSED = 1
    address = _messages.StringField(1)
    attachmentConfigurationConstraints = _messages.MessageField('InterconnectAttachmentConfigurationConstraints', 2)
    city = _messages.StringField(3)
    constraints = _messages.MessageField('InterconnectRemoteLocationConstraints', 4)
    continent = _messages.EnumField('ContinentValueValuesEnum', 5)
    creationTimestamp = _messages.StringField(6)
    description = _messages.StringField(7)
    facilityProvider = _messages.StringField(8)
    facilityProviderFacilityId = _messages.StringField(9)
    id = _messages.IntegerField(10, variant=_messages.Variant.UINT64)
    kind = _messages.StringField(11, default='compute#interconnectRemoteLocation')
    lacp = _messages.EnumField('LacpValueValuesEnum', 12)
    maxLagSize100Gbps = _messages.IntegerField(13, variant=_messages.Variant.INT32)
    maxLagSize10Gbps = _messages.IntegerField(14, variant=_messages.Variant.INT32)
    name = _messages.StringField(15)
    peeringdbFacilityId = _messages.StringField(16)
    permittedConnections = _messages.MessageField('InterconnectRemoteLocationPermittedConnections', 17, repeated=True)
    remoteService = _messages.StringField(18)
    selfLink = _messages.StringField(19)
    status = _messages.EnumField('StatusValueValuesEnum', 20)