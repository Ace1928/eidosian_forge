from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InterconnectLocation(_messages.Message):
    """Represents an Interconnect Attachment (VLAN) Location resource. You can
  use this resource to find location details about an Interconnect attachment
  (VLAN). For more information about interconnect attachments, read Creating
  VLAN Attachments.

  Enums:
    AvailableFeaturesValueListEntryValuesEnum:
    AvailableLinkTypesValueListEntryValuesEnum:
    ContinentValueValuesEnum: [Output Only] Continent for this location, which
      can take one of the following values: - AFRICA - ASIA_PAC - EUROPE -
      NORTH_AMERICA - SOUTH_AMERICA
    StatusValueValuesEnum: [Output Only] The status of this
      InterconnectLocation, which can take one of the following values: -
      CLOSED: The InterconnectLocation is closed and is unavailable for
      provisioning new Interconnects. - AVAILABLE: The InterconnectLocation is
      available for provisioning new Interconnects.

  Fields:
    address: [Output Only] The postal address of the Point of Presence, each
      line in the address is separated by a newline character.
    availabilityZone: [Output Only] Availability zone for this
      InterconnectLocation. Within a metropolitan area (metro), maintenance
      will not be simultaneously scheduled in more than one availability zone.
      Example: "zone1" or "zone2".
    availableFeatures: [Output only] List of features available at this
      InterconnectLocation, which can take one of the following values: -
      MACSEC
    availableLinkTypes: [Output only] List of link types available at this
      InterconnectLocation, which can take one of the following values: -
      LINK_TYPE_ETHERNET_10G_LR - LINK_TYPE_ETHERNET_100G_LR
    city: [Output Only] Metropolitan area designator that indicates which city
      an interconnect is located. For example: "Chicago, IL", "Amsterdam,
      Netherlands".
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
      compute#interconnectLocation for interconnect locations.
    name: [Output Only] Name of the resource.
    peeringdbFacilityId: [Output Only] The peeringdb identifier for this
      facility (corresponding with a netfac type in peeringdb).
    regionInfos: [Output Only] A list of InterconnectLocation.RegionInfo
      objects, that describe parameters pertaining to the relation between
      this InterconnectLocation and various Google Cloud regions.
    selfLink: [Output Only] Server-defined URL for the resource.
    status: [Output Only] The status of this InterconnectLocation, which can
      take one of the following values: - CLOSED: The InterconnectLocation is
      closed and is unavailable for provisioning new Interconnects. -
      AVAILABLE: The InterconnectLocation is available for provisioning new
      Interconnects.
    supportsPzs: [Output Only] Reserved for future use.
  """

    class AvailableFeaturesValueListEntryValuesEnum(_messages.Enum):
        """AvailableFeaturesValueListEntryValuesEnum enum type.

    Values:
      IF_MACSEC: Media Access Control security (MACsec)
    """
        IF_MACSEC = 0

    class AvailableLinkTypesValueListEntryValuesEnum(_messages.Enum):
        """AvailableLinkTypesValueListEntryValuesEnum enum type.

    Values:
      LINK_TYPE_ETHERNET_100G_LR: 100G Ethernet, LR Optics.
      LINK_TYPE_ETHERNET_10G_LR: 10G Ethernet, LR Optics. [(rate_bps) =
        10000000000];
    """
        LINK_TYPE_ETHERNET_100G_LR = 0
        LINK_TYPE_ETHERNET_10G_LR = 1

    class ContinentValueValuesEnum(_messages.Enum):
        """[Output Only] Continent for this location, which can take one of the
    following values: - AFRICA - ASIA_PAC - EUROPE - NORTH_AMERICA -
    SOUTH_AMERICA

    Values:
      AFRICA: <no description>
      ASIA_PAC: <no description>
      C_AFRICA: <no description>
      C_ASIA_PAC: <no description>
      C_EUROPE: <no description>
      C_NORTH_AMERICA: <no description>
      C_SOUTH_AMERICA: <no description>
      EUROPE: <no description>
      NORTH_AMERICA: <no description>
      SOUTH_AMERICA: <no description>
    """
        AFRICA = 0
        ASIA_PAC = 1
        C_AFRICA = 2
        C_ASIA_PAC = 3
        C_EUROPE = 4
        C_NORTH_AMERICA = 5
        C_SOUTH_AMERICA = 6
        EUROPE = 7
        NORTH_AMERICA = 8
        SOUTH_AMERICA = 9

    class StatusValueValuesEnum(_messages.Enum):
        """[Output Only] The status of this InterconnectLocation, which can take
    one of the following values: - CLOSED: The InterconnectLocation is closed
    and is unavailable for provisioning new Interconnects. - AVAILABLE: The
    InterconnectLocation is available for provisioning new Interconnects.

    Values:
      AVAILABLE: The InterconnectLocation is available for provisioning new
        Interconnects.
      CLOSED: The InterconnectLocation is closed for provisioning new
        Interconnects.
    """
        AVAILABLE = 0
        CLOSED = 1
    address = _messages.StringField(1)
    availabilityZone = _messages.StringField(2)
    availableFeatures = _messages.EnumField('AvailableFeaturesValueListEntryValuesEnum', 3, repeated=True)
    availableLinkTypes = _messages.EnumField('AvailableLinkTypesValueListEntryValuesEnum', 4, repeated=True)
    city = _messages.StringField(5)
    continent = _messages.EnumField('ContinentValueValuesEnum', 6)
    creationTimestamp = _messages.StringField(7)
    description = _messages.StringField(8)
    facilityProvider = _messages.StringField(9)
    facilityProviderFacilityId = _messages.StringField(10)
    id = _messages.IntegerField(11, variant=_messages.Variant.UINT64)
    kind = _messages.StringField(12, default='compute#interconnectLocation')
    name = _messages.StringField(13)
    peeringdbFacilityId = _messages.StringField(14)
    regionInfos = _messages.MessageField('InterconnectLocationRegionInfo', 15, repeated=True)
    selfLink = _messages.StringField(16)
    status = _messages.EnumField('StatusValueValuesEnum', 17)
    supportsPzs = _messages.BooleanField(18)