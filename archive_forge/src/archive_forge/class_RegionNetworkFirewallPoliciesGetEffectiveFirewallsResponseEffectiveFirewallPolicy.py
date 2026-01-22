from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RegionNetworkFirewallPoliciesGetEffectiveFirewallsResponseEffectiveFirewallPolicy(_messages.Message):
    """A RegionNetworkFirewallPoliciesGetEffectiveFirewallsResponseEffectiveFir
  ewallPolicy object.

  Enums:
    TypeValueValuesEnum: [Output Only] The type of the firewall policy. Can be
      one of HIERARCHY, NETWORK, NETWORK_REGIONAL, SYSTEM_GLOBAL,
      SYSTEM_REGIONAL.

  Fields:
    displayName: [Output Only] The display name of the firewall policy.
    name: [Output Only] The name of the firewall policy.
    rules: The rules that apply to the network.
    type: [Output Only] The type of the firewall policy. Can be one of
      HIERARCHY, NETWORK, NETWORK_REGIONAL, SYSTEM_GLOBAL, SYSTEM_REGIONAL.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """[Output Only] The type of the firewall policy. Can be one of
    HIERARCHY, NETWORK, NETWORK_REGIONAL, SYSTEM_GLOBAL, SYSTEM_REGIONAL.

    Values:
      HIERARCHY: <no description>
      NETWORK: <no description>
      NETWORK_REGIONAL: <no description>
      UNSPECIFIED: <no description>
    """
        HIERARCHY = 0
        NETWORK = 1
        NETWORK_REGIONAL = 2
        UNSPECIFIED = 3
    displayName = _messages.StringField(1)
    name = _messages.StringField(2)
    rules = _messages.MessageField('FirewallPolicyRule', 3, repeated=True)
    type = _messages.EnumField('TypeValueValuesEnum', 4)