from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirewallPolicyRuleMatcher(_messages.Message):
    """Represents a match condition that incoming traffic is evaluated against.
  Exactly one field must be specified.

  Fields:
    destAddressGroups: Address groups which should be matched against the
      traffic destination. Maximum number of destination address groups is 10.
    destFqdns: Fully Qualified Domain Name (FQDN) which should be matched
      against traffic destination. Maximum number of destination fqdn allowed
      is 100.
    destIpRanges: CIDR IP address range. Maximum number of destination CIDR IP
      ranges allowed is 5000.
    destRegionCodes: Region codes whose IP addresses will be used to match for
      destination of traffic. Should be specified as 2 letter country code
      defined as per ISO 3166 alpha-2 country codes. ex."US" Maximum number of
      dest region codes allowed is 5000.
    destThreatIntelligences: Names of Network Threat Intelligence lists. The
      IPs in these lists will be matched against traffic destination.
    layer4Configs: Pairs of IP protocols and ports that the rule should match.
    srcAddressGroups: Address groups which should be matched against the
      traffic source. Maximum number of source address groups is 10.
    srcFqdns: Fully Qualified Domain Name (FQDN) which should be matched
      against traffic source. Maximum number of source fqdn allowed is 100.
    srcIpRanges: CIDR IP address range. Maximum number of source CIDR IP
      ranges allowed is 5000.
    srcRegionCodes: Region codes whose IP addresses will be used to match for
      source of traffic. Should be specified as 2 letter country code defined
      as per ISO 3166 alpha-2 country codes. ex."US" Maximum number of source
      region codes allowed is 5000.
    srcSecureTags: List of secure tag values, which should be matched at the
      source of the traffic. For INGRESS rule, if all the srcSecureTag are
      INEFFECTIVE, and there is no srcIpRange, this rule will be ignored.
      Maximum number of source tag values allowed is 256.
    srcThreatIntelligences: Names of Network Threat Intelligence lists. The
      IPs in these lists will be matched against traffic source.
  """
    destAddressGroups = _messages.StringField(1, repeated=True)
    destFqdns = _messages.StringField(2, repeated=True)
    destIpRanges = _messages.StringField(3, repeated=True)
    destRegionCodes = _messages.StringField(4, repeated=True)
    destThreatIntelligences = _messages.StringField(5, repeated=True)
    layer4Configs = _messages.MessageField('FirewallPolicyRuleMatcherLayer4Config', 6, repeated=True)
    srcAddressGroups = _messages.StringField(7, repeated=True)
    srcFqdns = _messages.StringField(8, repeated=True)
    srcIpRanges = _messages.StringField(9, repeated=True)
    srcRegionCodes = _messages.StringField(10, repeated=True)
    srcSecureTags = _messages.MessageField('FirewallPolicyRuleSecureTag', 11, repeated=True)
    srcThreatIntelligences = _messages.StringField(12, repeated=True)