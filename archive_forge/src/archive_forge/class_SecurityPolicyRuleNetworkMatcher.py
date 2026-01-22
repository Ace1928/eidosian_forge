from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityPolicyRuleNetworkMatcher(_messages.Message):
    """Represents a match condition that incoming network traffic is evaluated
  against.

  Fields:
    destIpRanges: Destination IPv4/IPv6 addresses or CIDR prefixes, in
      standard text format.
    destPorts: Destination port numbers for TCP/UDP/SCTP. Each element can be
      a 16-bit unsigned decimal number (e.g. "80") or range (e.g. "0-1023").
    ipProtocols: IPv4 protocol / IPv6 next header (after extension headers).
      Each element can be an 8-bit unsigned decimal number (e.g. "6"), range
      (e.g. "253-254"), or one of the following protocol names: "tcp", "udp",
      "icmp", "esp", "ah", "ipip", or "sctp".
    srcAsns: BGP Autonomous System Number associated with the source IP
      address.
    srcIpRanges: Source IPv4/IPv6 addresses or CIDR prefixes, in standard text
      format.
    srcPorts: Source port numbers for TCP/UDP/SCTP. Each element can be a
      16-bit unsigned decimal number (e.g. "80") or range (e.g. "0-1023").
    srcRegionCodes: Two-letter ISO 3166-1 alpha-2 country code associated with
      the source IP address.
    userDefinedFields: User-defined fields. Each element names a defined field
      and lists the matching values for that field.
  """
    destIpRanges = _messages.StringField(1, repeated=True)
    destPorts = _messages.StringField(2, repeated=True)
    ipProtocols = _messages.StringField(3, repeated=True)
    srcAsns = _messages.IntegerField(4, repeated=True, variant=_messages.Variant.UINT32)
    srcIpRanges = _messages.StringField(5, repeated=True)
    srcPorts = _messages.StringField(6, repeated=True)
    srcRegionCodes = _messages.StringField(7, repeated=True)
    userDefinedFields = _messages.MessageField('SecurityPolicyRuleNetworkMatcherUserDefinedFieldMatch', 8, repeated=True)