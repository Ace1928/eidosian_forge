from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ManagedZoneForwardingConfigNameServerTarget(_messages.Message):
    """A ManagedZoneForwardingConfigNameServerTarget object.

  Enums:
    ForwardingPathValueValuesEnum: Forwarding path for this NameServerTarget.
      If unset or set to DEFAULT, Cloud DNS makes forwarding decisions based
      on IP address ranges; that is, RFC1918 addresses go to the VPC network,
      non-RFC1918 addresses go to the internet. When set to PRIVATE, Cloud DNS
      always sends queries through the VPC network for this target.

  Fields:
    forwardingPath: Forwarding path for this NameServerTarget. If unset or set
      to DEFAULT, Cloud DNS makes forwarding decisions based on IP address
      ranges; that is, RFC1918 addresses go to the VPC network, non-RFC1918
      addresses go to the internet. When set to PRIVATE, Cloud DNS always
      sends queries through the VPC network for this target.
    ipv4Address: IPv4 address of a target name server.
    ipv6Address: IPv6 address of a target name server. Does not accept both
      fields (ipv4 & ipv6) being populated. Public preview as of November
      2022.
    kind: A string attribute.
  """

    class ForwardingPathValueValuesEnum(_messages.Enum):
        """Forwarding path for this NameServerTarget. If unset or set to DEFAULT,
    Cloud DNS makes forwarding decisions based on IP address ranges; that is,
    RFC1918 addresses go to the VPC network, non-RFC1918 addresses go to the
    internet. When set to PRIVATE, Cloud DNS always sends queries through the
    VPC network for this target.

    Values:
      default: Cloud DNS makes forwarding decisions based on address ranges;
        that is, RFC1918 addresses forward to the target through the VPC and
        non-RFC1918 addresses forward to the target through the internet
      private: Cloud DNS always forwards to this target through the VPC.
    """
        default = 0
        private = 1
    forwardingPath = _messages.EnumField('ForwardingPathValueValuesEnum', 1)
    ipv4Address = _messages.StringField(2)
    ipv6Address = _messages.StringField(3)
    kind = _messages.StringField(4, default='dns#managedZoneForwardingConfigNameServerTarget')