from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InterconnectAttachment(_messages.Message):
    """Represents an Interconnect Attachment (VLAN) resource. You can use
  Interconnect attachments (VLANS) to connect your Virtual Private Cloud
  networks to your on-premises networks through an Interconnect. For more
  information, read Creating VLAN Attachments.

  Enums:
    BandwidthValueValuesEnum: Provisioned bandwidth capacity for the
      interconnect attachment. For attachments of type DEDICATED, the user can
      set the bandwidth. For attachments of type PARTNER, the Google Partner
      that is operating the interconnect must set the bandwidth. Output only
      for PARTNER type, mutable for PARTNER_PROVIDER and DEDICATED, and can
      take one of the following values: - BPS_50M: 50 Mbit/s - BPS_100M: 100
      Mbit/s - BPS_200M: 200 Mbit/s - BPS_300M: 300 Mbit/s - BPS_400M: 400
      Mbit/s - BPS_500M: 500 Mbit/s - BPS_1G: 1 Gbit/s - BPS_2G: 2 Gbit/s -
      BPS_5G: 5 Gbit/s - BPS_10G: 10 Gbit/s - BPS_20G: 20 Gbit/s - BPS_50G: 50
      Gbit/s
    EdgeAvailabilityDomainValueValuesEnum: Desired availability domain for the
      attachment. Only available for type PARTNER, at creation time, and can
      take one of the following values: - AVAILABILITY_DOMAIN_ANY -
      AVAILABILITY_DOMAIN_1 - AVAILABILITY_DOMAIN_2 For improved reliability,
      customers should configure a pair of attachments, one per availability
      domain. The selected availability domain will be provided to the Partner
      via the pairing key, so that the provisioned circuit will lie in the
      specified domain. If not specified, the value will default to
      AVAILABILITY_DOMAIN_ANY.
    EncryptionValueValuesEnum: Indicates the user-supplied encryption option
      of this VLAN attachment (interconnectAttachment). Can only be specified
      at attachment creation for PARTNER or DEDICATED attachments. Possible
      values are: - NONE - This is the default value, which means that the
      VLAN attachment carries unencrypted traffic. VMs are able to send
      traffic to, or receive traffic from, such a VLAN attachment. - IPSEC -
      The VLAN attachment carries only encrypted traffic that is encrypted by
      an IPsec device, such as an HA VPN gateway or third-party IPsec VPN. VMs
      cannot directly send traffic to, or receive traffic from, such a VLAN
      attachment. To use *HA VPN over Cloud Interconnect*, the VLAN attachment
      must be created with this option.
    OperationalStatusValueValuesEnum: [Output Only] The current status of
      whether or not this interconnect attachment is functional, which can
      take one of the following values: - OS_ACTIVE: The attachment has been
      turned up and is ready to use. - OS_UNPROVISIONED: The attachment is not
      ready to use yet, because turnup is not complete.
    StackTypeValueValuesEnum: The stack type for this interconnect attachment
      to identify whether the IPv6 feature is enabled or not. If not
      specified, IPV4_ONLY will be used. This field can be both set at
      interconnect attachments creation and update interconnect attachment
      operations.
    StateValueValuesEnum: [Output Only] The current state of this attachment's
      functionality. Enum values ACTIVE and UNPROVISIONED are shared by
      DEDICATED/PRIVATE, PARTNER, and PARTNER_PROVIDER interconnect
      attachments, while enum values PENDING_PARTNER,
      PARTNER_REQUEST_RECEIVED, and PENDING_CUSTOMER are used for only PARTNER
      and PARTNER_PROVIDER interconnect attachments. This state can take one
      of the following values: - ACTIVE: The attachment has been turned up and
      is ready to use. - UNPROVISIONED: The attachment is not ready to use
      yet, because turnup is not complete. - PENDING_PARTNER: A newly-created
      PARTNER attachment that has not yet been configured on the Partner side.
      - PARTNER_REQUEST_RECEIVED: A PARTNER attachment is in the process of
      provisioning after a PARTNER_PROVIDER attachment was created that
      references it. - PENDING_CUSTOMER: A PARTNER or PARTNER_PROVIDER
      attachment that is waiting for a customer to activate it. - DEFUNCT: The
      attachment was deleted externally and is no longer functional. This
      could be because the associated Interconnect was removed, or because the
      other side of a Partner attachment was deleted.
    TypeValueValuesEnum: The type of interconnect attachment this is, which
      can take one of the following values: - DEDICATED: an attachment to a
      Dedicated Interconnect. - PARTNER: an attachment to a Partner
      Interconnect, created by the customer. - PARTNER_PROVIDER: an attachment
      to a Partner Interconnect, created by the partner.

  Messages:
    LabelsValue: Labels for this resource. These can only be added or modified
      by the setLabels method. Each label key/value pair must comply with
      RFC1035. Label values may be empty.

  Fields:
    adminEnabled: Determines whether this Attachment will carry packets. Not
      present for PARTNER_PROVIDER.
    bandwidth: Provisioned bandwidth capacity for the interconnect attachment.
      For attachments of type DEDICATED, the user can set the bandwidth. For
      attachments of type PARTNER, the Google Partner that is operating the
      interconnect must set the bandwidth. Output only for PARTNER type,
      mutable for PARTNER_PROVIDER and DEDICATED, and can take one of the
      following values: - BPS_50M: 50 Mbit/s - BPS_100M: 100 Mbit/s -
      BPS_200M: 200 Mbit/s - BPS_300M: 300 Mbit/s - BPS_400M: 400 Mbit/s -
      BPS_500M: 500 Mbit/s - BPS_1G: 1 Gbit/s - BPS_2G: 2 Gbit/s - BPS_5G: 5
      Gbit/s - BPS_10G: 10 Gbit/s - BPS_20G: 20 Gbit/s - BPS_50G: 50 Gbit/s
    candidateIpv6Subnets: This field is not available.
    candidateSubnets: Up to 16 candidate prefixes that can be used to restrict
      the allocation of cloudRouterIpAddress and customerRouterIpAddress for
      this attachment. All prefixes must be within link-local address space
      (169.254.0.0/16) and must be /29 or shorter (/28, /27, etc). Google will
      attempt to select an unused /29 from the supplied candidate prefix(es).
      The request will fail if all possible /29s are in use on Google's edge.
      If not supplied, Google will randomly select an unused /29 from all of
      link-local space.
    cloudRouterIpAddress: [Output Only] IPv4 address + prefix length to be
      configured on Cloud Router Interface for this interconnect attachment.
    cloudRouterIpv6Address: [Output Only] IPv6 address + prefix length to be
      configured on Cloud Router Interface for this interconnect attachment.
    cloudRouterIpv6InterfaceId: This field is not available.
    configurationConstraints: [Output Only] Constraints for this attachment,
      if any. The attachment does not work if these constraints are not met.
    creationTimestamp: [Output Only] Creation timestamp in RFC3339 text
      format.
    customerRouterIpAddress: [Output Only] IPv4 address + prefix length to be
      configured on the customer router subinterface for this interconnect
      attachment.
    customerRouterIpv6Address: [Output Only] IPv6 address + prefix length to
      be configured on the customer router subinterface for this interconnect
      attachment.
    customerRouterIpv6InterfaceId: This field is not available.
    dataplaneVersion: [Output Only] Dataplane version for this
      InterconnectAttachment. This field is only present for Dataplane version
      2 and higher. Absence of this field in the API output indicates that the
      Dataplane is version 1.
    description: An optional description of this resource.
    edgeAvailabilityDomain: Desired availability domain for the attachment.
      Only available for type PARTNER, at creation time, and can take one of
      the following values: - AVAILABILITY_DOMAIN_ANY - AVAILABILITY_DOMAIN_1
      - AVAILABILITY_DOMAIN_2 For improved reliability, customers should
      configure a pair of attachments, one per availability domain. The
      selected availability domain will be provided to the Partner via the
      pairing key, so that the provisioned circuit will lie in the specified
      domain. If not specified, the value will default to
      AVAILABILITY_DOMAIN_ANY.
    encryption: Indicates the user-supplied encryption option of this VLAN
      attachment (interconnectAttachment). Can only be specified at attachment
      creation for PARTNER or DEDICATED attachments. Possible values are: -
      NONE - This is the default value, which means that the VLAN attachment
      carries unencrypted traffic. VMs are able to send traffic to, or receive
      traffic from, such a VLAN attachment. - IPSEC - The VLAN attachment
      carries only encrypted traffic that is encrypted by an IPsec device,
      such as an HA VPN gateway or third-party IPsec VPN. VMs cannot directly
      send traffic to, or receive traffic from, such a VLAN attachment. To use
      *HA VPN over Cloud Interconnect*, the VLAN attachment must be created
      with this option.
    googleReferenceId: [Output Only] Google reference ID, to be used when
      raising support tickets with Google or otherwise to debug backend
      connectivity issues. [Deprecated] This field is not used.
    id: [Output Only] The unique identifier for the resource. This identifier
      is defined by the server.
    interconnect: URL of the underlying Interconnect object that this
      attachment's traffic will traverse through.
    ipsecInternalAddresses: A list of URLs of addresses that have been
      reserved for the VLAN attachment. Used only for the VLAN attachment that
      has the encryption option as IPSEC. The addresses must be regional
      internal IP address ranges. When creating an HA VPN gateway over the
      VLAN attachment, if the attachment is configured to use a regional
      internal IP address, then the VPN gateway's IP address is allocated from
      the IP address range specified here. For example, if the HA VPN
      gateway's interface 0 is paired to this VLAN attachment, then a regional
      internal IP address for the VPN gateway interface 0 will be allocated
      from the IP address specified for this VLAN attachment. If this field is
      not specified when creating the VLAN attachment, then later on when
      creating an HA VPN gateway on this VLAN attachment, the HA VPN gateway's
      IP address is allocated from the regional external IP address pool.
    kind: [Output Only] Type of the resource. Always
      compute#interconnectAttachment for interconnect attachments.
    labelFingerprint: A fingerprint for the labels being applied to this
      InterconnectAttachment, which is essentially a hash of the labels set
      used for optimistic locking. The fingerprint is initially generated by
      Compute Engine and changes after every request to modify or update
      labels. You must always provide an up-to-date fingerprint hash in order
      to update or change labels, otherwise the request will fail with error
      412 conditionNotMet. To see the latest fingerprint, make a get() request
      to retrieve an InterconnectAttachment.
    labels: Labels for this resource. These can only be added or modified by
      the setLabels method. Each label key/value pair must comply with
      RFC1035. Label values may be empty.
    mtu: Maximum Transmission Unit (MTU), in bytes, of packets passing through
      this interconnect attachment. Only 1440 and 1500 are allowed. If not
      specified, the value will default to 1440.
    name: Name of the resource. Provided by the client when the resource is
      created. The name must be 1-63 characters long, and comply with RFC1035.
      Specifically, the name must be 1-63 characters long and match the
      regular expression `[a-z]([-a-z0-9]*[a-z0-9])?` which means the first
      character must be a lowercase letter, and all following characters must
      be a dash, lowercase letter, or digit, except the last character, which
      cannot be a dash.
    operationalStatus: [Output Only] The current status of whether or not this
      interconnect attachment is functional, which can take one of the
      following values: - OS_ACTIVE: The attachment has been turned up and is
      ready to use. - OS_UNPROVISIONED: The attachment is not ready to use
      yet, because turnup is not complete.
    pairingKey: [Output only for type PARTNER. Input only for
      PARTNER_PROVIDER. Not present for DEDICATED]. The opaque identifier of a
      PARTNER attachment used to initiate provisioning with a selected
      partner. Of the form "XXXXX/region/domain"
    partnerAsn: Optional BGP ASN for the router supplied by a Layer 3 Partner
      if they configured BGP on behalf of the customer. Output only for
      PARTNER type, input only for PARTNER_PROVIDER, not available for
      DEDICATED.
    partnerMetadata: Informational metadata about Partner attachments from
      Partners to display to customers. Output only for PARTNER type, mutable
      for PARTNER_PROVIDER, not available for DEDICATED.
    privateInterconnectInfo: [Output Only] Information specific to an
      InterconnectAttachment. This property is populated if the interconnect
      that this is attached to is of type DEDICATED.
    region: [Output Only] URL of the region where the regional interconnect
      attachment resides. You must specify this field as part of the HTTP
      request URL. It is not settable as a field in the request body.
    remoteService: [Output Only] If the attachment is on a Cross-Cloud
      Interconnect connection, this field contains the interconnect's remote
      location service provider. Example values: "Amazon Web Services"
      "Microsoft Azure". The field is set only for attachments on Cross-Cloud
      Interconnect connections. Its value is copied from the
      InterconnectRemoteLocation remoteService field.
    router: URL of the Cloud Router to be used for dynamic routing. This
      router must be in the same region as this InterconnectAttachment. The
      InterconnectAttachment will automatically connect the Interconnect to
      the network & region within which the Cloud Router is configured.
    satisfiesPzs: [Output Only] Reserved for future use.
    selfLink: [Output Only] Server-defined URL for the resource.
    stackType: The stack type for this interconnect attachment to identify
      whether the IPv6 feature is enabled or not. If not specified, IPV4_ONLY
      will be used. This field can be both set at interconnect attachments
      creation and update interconnect attachment operations.
    state: [Output Only] The current state of this attachment's functionality.
      Enum values ACTIVE and UNPROVISIONED are shared by DEDICATED/PRIVATE,
      PARTNER, and PARTNER_PROVIDER interconnect attachments, while enum
      values PENDING_PARTNER, PARTNER_REQUEST_RECEIVED, and PENDING_CUSTOMER
      are used for only PARTNER and PARTNER_PROVIDER interconnect attachments.
      This state can take one of the following values: - ACTIVE: The
      attachment has been turned up and is ready to use. - UNPROVISIONED: The
      attachment is not ready to use yet, because turnup is not complete. -
      PENDING_PARTNER: A newly-created PARTNER attachment that has not yet
      been configured on the Partner side. - PARTNER_REQUEST_RECEIVED: A
      PARTNER attachment is in the process of provisioning after a
      PARTNER_PROVIDER attachment was created that references it. -
      PENDING_CUSTOMER: A PARTNER or PARTNER_PROVIDER attachment that is
      waiting for a customer to activate it. - DEFUNCT: The attachment was
      deleted externally and is no longer functional. This could be because
      the associated Interconnect was removed, or because the other side of a
      Partner attachment was deleted.
    subnetLength: Length of the IPv4 subnet mask. Allowed values: - 29
      (default) - 30 The default value is 29, except for Cross-Cloud
      Interconnect connections that use an InterconnectRemoteLocation with a
      constraints.subnetLengthRange.min equal to 30. For example, connections
      that use an Azure remote location fall into this category. In these
      cases, the default value is 30, and requesting 29 returns an error.
      Where both 29 and 30 are allowed, 29 is preferred, because it gives
      Google Cloud Support more debugging visibility.
    type: The type of interconnect attachment this is, which can take one of
      the following values: - DEDICATED: an attachment to a Dedicated
      Interconnect. - PARTNER: an attachment to a Partner Interconnect,
      created by the customer. - PARTNER_PROVIDER: an attachment to a Partner
      Interconnect, created by the partner.
    vlanTag8021q: The IEEE 802.1Q VLAN tag for this attachment, in the range
      2-4093. Only specified at creation time.
  """

    class BandwidthValueValuesEnum(_messages.Enum):
        """Provisioned bandwidth capacity for the interconnect attachment. For
    attachments of type DEDICATED, the user can set the bandwidth. For
    attachments of type PARTNER, the Google Partner that is operating the
    interconnect must set the bandwidth. Output only for PARTNER type, mutable
    for PARTNER_PROVIDER and DEDICATED, and can take one of the following
    values: - BPS_50M: 50 Mbit/s - BPS_100M: 100 Mbit/s - BPS_200M: 200 Mbit/s
    - BPS_300M: 300 Mbit/s - BPS_400M: 400 Mbit/s - BPS_500M: 500 Mbit/s -
    BPS_1G: 1 Gbit/s - BPS_2G: 2 Gbit/s - BPS_5G: 5 Gbit/s - BPS_10G: 10
    Gbit/s - BPS_20G: 20 Gbit/s - BPS_50G: 50 Gbit/s

    Values:
      BPS_100M: 100 Mbit/s
      BPS_10G: 10 Gbit/s
      BPS_1G: 1 Gbit/s
      BPS_200M: 200 Mbit/s
      BPS_20G: 20 Gbit/s
      BPS_2G: 2 Gbit/s
      BPS_300M: 300 Mbit/s
      BPS_400M: 400 Mbit/s
      BPS_500M: 500 Mbit/s
      BPS_50G: 50 Gbit/s
      BPS_50M: 50 Mbit/s
      BPS_5G: 5 Gbit/s
    """
        BPS_100M = 0
        BPS_10G = 1
        BPS_1G = 2
        BPS_200M = 3
        BPS_20G = 4
        BPS_2G = 5
        BPS_300M = 6
        BPS_400M = 7
        BPS_500M = 8
        BPS_50G = 9
        BPS_50M = 10
        BPS_5G = 11

    class EdgeAvailabilityDomainValueValuesEnum(_messages.Enum):
        """Desired availability domain for the attachment. Only available for
    type PARTNER, at creation time, and can take one of the following values:
    - AVAILABILITY_DOMAIN_ANY - AVAILABILITY_DOMAIN_1 - AVAILABILITY_DOMAIN_2
    For improved reliability, customers should configure a pair of
    attachments, one per availability domain. The selected availability domain
    will be provided to the Partner via the pairing key, so that the
    provisioned circuit will lie in the specified domain. If not specified,
    the value will default to AVAILABILITY_DOMAIN_ANY.

    Values:
      AVAILABILITY_DOMAIN_1: <no description>
      AVAILABILITY_DOMAIN_2: <no description>
      AVAILABILITY_DOMAIN_ANY: <no description>
    """
        AVAILABILITY_DOMAIN_1 = 0
        AVAILABILITY_DOMAIN_2 = 1
        AVAILABILITY_DOMAIN_ANY = 2

    class EncryptionValueValuesEnum(_messages.Enum):
        """Indicates the user-supplied encryption option of this VLAN attachment
    (interconnectAttachment). Can only be specified at attachment creation for
    PARTNER or DEDICATED attachments. Possible values are: - NONE - This is
    the default value, which means that the VLAN attachment carries
    unencrypted traffic. VMs are able to send traffic to, or receive traffic
    from, such a VLAN attachment. - IPSEC - The VLAN attachment carries only
    encrypted traffic that is encrypted by an IPsec device, such as an HA VPN
    gateway or third-party IPsec VPN. VMs cannot directly send traffic to, or
    receive traffic from, such a VLAN attachment. To use *HA VPN over Cloud
    Interconnect*, the VLAN attachment must be created with this option.

    Values:
      IPSEC: The interconnect attachment will carry only encrypted traffic
        that is encrypted by an IPsec device such as HA VPN gateway; VMs
        cannot directly send traffic to or receive traffic from such an
        interconnect attachment. To use HA VPN over Cloud Interconnect, the
        interconnect attachment must be created with this option.
      NONE: This is the default value, which means the Interconnect Attachment
        will carry unencrypted traffic. VMs will be able to send traffic to or
        receive traffic from such interconnect attachment.
    """
        IPSEC = 0
        NONE = 1

    class OperationalStatusValueValuesEnum(_messages.Enum):
        """[Output Only] The current status of whether or not this interconnect
    attachment is functional, which can take one of the following values: -
    OS_ACTIVE: The attachment has been turned up and is ready to use. -
    OS_UNPROVISIONED: The attachment is not ready to use yet, because turnup
    is not complete.

    Values:
      OS_ACTIVE: Indicates that attachment has been turned up and is ready to
        use.
      OS_UNPROVISIONED: Indicates that attachment is not ready to use yet,
        because turnup is not complete.
    """
        OS_ACTIVE = 0
        OS_UNPROVISIONED = 1

    class StackTypeValueValuesEnum(_messages.Enum):
        """The stack type for this interconnect attachment to identify whether
    the IPv6 feature is enabled or not. If not specified, IPV4_ONLY will be
    used. This field can be both set at interconnect attachments creation and
    update interconnect attachment operations.

    Values:
      IPV4_IPV6: The interconnect attachment can have both IPv4 and IPv6
        addresses.
      IPV4_ONLY: The interconnect attachment will only be assigned IPv4
        addresses.
    """
        IPV4_IPV6 = 0
        IPV4_ONLY = 1

    class StateValueValuesEnum(_messages.Enum):
        """[Output Only] The current state of this attachment's functionality.
    Enum values ACTIVE and UNPROVISIONED are shared by DEDICATED/PRIVATE,
    PARTNER, and PARTNER_PROVIDER interconnect attachments, while enum values
    PENDING_PARTNER, PARTNER_REQUEST_RECEIVED, and PENDING_CUSTOMER are used
    for only PARTNER and PARTNER_PROVIDER interconnect attachments. This state
    can take one of the following values: - ACTIVE: The attachment has been
    turned up and is ready to use. - UNPROVISIONED: The attachment is not
    ready to use yet, because turnup is not complete. - PENDING_PARTNER: A
    newly-created PARTNER attachment that has not yet been configured on the
    Partner side. - PARTNER_REQUEST_RECEIVED: A PARTNER attachment is in the
    process of provisioning after a PARTNER_PROVIDER attachment was created
    that references it. - PENDING_CUSTOMER: A PARTNER or PARTNER_PROVIDER
    attachment that is waiting for a customer to activate it. - DEFUNCT: The
    attachment was deleted externally and is no longer functional. This could
    be because the associated Interconnect was removed, or because the other
    side of a Partner attachment was deleted.

    Values:
      ACTIVE: Indicates that attachment has been turned up and is ready to
        use.
      DEFUNCT: The attachment was deleted externally and is no longer
        functional. This could be because the associated Interconnect was
        wiped out, or because the other side of a Partner attachment was
        deleted.
      PARTNER_REQUEST_RECEIVED: A PARTNER attachment is in the process of
        provisioning after a PARTNER_PROVIDER attachment was created that
        references it.
      PENDING_CUSTOMER: PARTNER or PARTNER_PROVIDER attachment that is waiting
        for the customer to activate.
      PENDING_PARTNER: A newly created PARTNER attachment that has not yet
        been configured on the Partner side.
      STATE_UNSPECIFIED: <no description>
      UNPROVISIONED: Indicates that attachment is not ready to use yet,
        because turnup is not complete.
    """
        ACTIVE = 0
        DEFUNCT = 1
        PARTNER_REQUEST_RECEIVED = 2
        PENDING_CUSTOMER = 3
        PENDING_PARTNER = 4
        STATE_UNSPECIFIED = 5
        UNPROVISIONED = 6

    class TypeValueValuesEnum(_messages.Enum):
        """The type of interconnect attachment this is, which can take one of the
    following values: - DEDICATED: an attachment to a Dedicated Interconnect.
    - PARTNER: an attachment to a Partner Interconnect, created by the
    customer. - PARTNER_PROVIDER: an attachment to a Partner Interconnect,
    created by the partner.

    Values:
      DEDICATED: Attachment to a dedicated interconnect.
      PARTNER: Attachment to a partner interconnect, created by the customer.
      PARTNER_PROVIDER: Attachment to a partner interconnect, created by the
        partner.
    """
        DEDICATED = 0
        PARTNER = 1
        PARTNER_PROVIDER = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels for this resource. These can only be added or modified by the
    setLabels method. Each label key/value pair must comply with RFC1035.
    Label values may be empty.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    adminEnabled = _messages.BooleanField(1)
    bandwidth = _messages.EnumField('BandwidthValueValuesEnum', 2)
    candidateIpv6Subnets = _messages.StringField(3, repeated=True)
    candidateSubnets = _messages.StringField(4, repeated=True)
    cloudRouterIpAddress = _messages.StringField(5)
    cloudRouterIpv6Address = _messages.StringField(6)
    cloudRouterIpv6InterfaceId = _messages.StringField(7)
    configurationConstraints = _messages.MessageField('InterconnectAttachmentConfigurationConstraints', 8)
    creationTimestamp = _messages.StringField(9)
    customerRouterIpAddress = _messages.StringField(10)
    customerRouterIpv6Address = _messages.StringField(11)
    customerRouterIpv6InterfaceId = _messages.StringField(12)
    dataplaneVersion = _messages.IntegerField(13, variant=_messages.Variant.INT32)
    description = _messages.StringField(14)
    edgeAvailabilityDomain = _messages.EnumField('EdgeAvailabilityDomainValueValuesEnum', 15)
    encryption = _messages.EnumField('EncryptionValueValuesEnum', 16)
    googleReferenceId = _messages.StringField(17)
    id = _messages.IntegerField(18, variant=_messages.Variant.UINT64)
    interconnect = _messages.StringField(19)
    ipsecInternalAddresses = _messages.StringField(20, repeated=True)
    kind = _messages.StringField(21, default='compute#interconnectAttachment')
    labelFingerprint = _messages.BytesField(22)
    labels = _messages.MessageField('LabelsValue', 23)
    mtu = _messages.IntegerField(24, variant=_messages.Variant.INT32)
    name = _messages.StringField(25)
    operationalStatus = _messages.EnumField('OperationalStatusValueValuesEnum', 26)
    pairingKey = _messages.StringField(27)
    partnerAsn = _messages.IntegerField(28)
    partnerMetadata = _messages.MessageField('InterconnectAttachmentPartnerMetadata', 29)
    privateInterconnectInfo = _messages.MessageField('InterconnectAttachmentPrivateInfo', 30)
    region = _messages.StringField(31)
    remoteService = _messages.StringField(32)
    router = _messages.StringField(33)
    satisfiesPzs = _messages.BooleanField(34)
    selfLink = _messages.StringField(35)
    stackType = _messages.EnumField('StackTypeValueValuesEnum', 36)
    state = _messages.EnumField('StateValueValuesEnum', 37)
    subnetLength = _messages.IntegerField(38, variant=_messages.Variant.INT32)
    type = _messages.EnumField('TypeValueValuesEnum', 39)
    vlanTag8021q = _messages.IntegerField(40, variant=_messages.Variant.INT32)