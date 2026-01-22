from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RegionNetworkFirewallPoliciesGetEffectiveFirewallsResponse(_messages.Message):
    """A RegionNetworkFirewallPoliciesGetEffectiveFirewallsResponse object.

  Fields:
    firewallPolicys: Effective firewalls from firewall policy.
    firewalls: Effective firewalls on the network.
  """
    firewallPolicys = _messages.MessageField('RegionNetworkFirewallPoliciesGetEffectiveFirewallsResponseEffectiveFirewallPolicy', 1, repeated=True)
    firewalls = _messages.MessageField('Firewall', 2, repeated=True)