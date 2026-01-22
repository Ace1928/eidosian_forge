from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccessConfig(_messages.Message):
    """An access configuration attached to an instance's network interface.
  Only one access config per instance is supported.

  Enums:
    NetworkTierValueValuesEnum: Optional. This signifies the networking tier
      used for configuring this access
    TypeValueValuesEnum: Optional. In accessConfigs (IPv4), the default and
      only option is ONE_TO_ONE_NAT. In ipv6AccessConfigs, the default and
      only option is DIRECT_IPV6.

  Fields:
    externalIpv6: Optional. The external IPv6 address of this access
      configuration.
    externalIpv6PrefixLength: Optional. The prefix length of the external IPv6
      range.
    name: Optional. The name of this access configuration.
    natIP: Optional. The external IP address of this access configuration.
    networkTier: Optional. This signifies the networking tier used for
      configuring this access
    publicPtrDomainName: Optional. The DNS domain name for the public PTR
      record.
    setPublicPtr: Optional. Specifies whether a public DNS 'PTR' record should
      be created to map the external IP address of the instance to a DNS
      domain name.
    type: Optional. In accessConfigs (IPv4), the default and only option is
      ONE_TO_ONE_NAT. In ipv6AccessConfigs, the default and only option is
      DIRECT_IPV6.
  """

    class NetworkTierValueValuesEnum(_messages.Enum):
        """Optional. This signifies the networking tier used for configuring this
    access

    Values:
      NETWORK_TIER_UNSPECIFIED: Default value. This value is unused.
      PREMIUM: High quality, Google-grade network tier, support for all
        networking products.
      STANDARD: Public internet quality, only limited support for other
        networking products.
    """
        NETWORK_TIER_UNSPECIFIED = 0
        PREMIUM = 1
        STANDARD = 2

    class TypeValueValuesEnum(_messages.Enum):
        """Optional. In accessConfigs (IPv4), the default and only option is
    ONE_TO_ONE_NAT. In ipv6AccessConfigs, the default and only option is
    DIRECT_IPV6.

    Values:
      ACCESS_TYPE_UNSPECIFIED: Default value. This value is unused.
      ONE_TO_ONE_NAT: ONE_TO_ONE_NAT
      DIRECT_IPV6: Direct IPv6 access.
    """
        ACCESS_TYPE_UNSPECIFIED = 0
        ONE_TO_ONE_NAT = 1
        DIRECT_IPV6 = 2
    externalIpv6 = _messages.StringField(1)
    externalIpv6PrefixLength = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    name = _messages.StringField(3)
    natIP = _messages.StringField(4)
    networkTier = _messages.EnumField('NetworkTierValueValuesEnum', 5)
    publicPtrDomainName = _messages.StringField(6)
    setPublicPtr = _messages.BooleanField(7)
    type = _messages.EnumField('TypeValueValuesEnum', 8)