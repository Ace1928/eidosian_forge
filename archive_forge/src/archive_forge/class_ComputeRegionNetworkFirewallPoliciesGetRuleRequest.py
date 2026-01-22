from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeRegionNetworkFirewallPoliciesGetRuleRequest(_messages.Message):
    """A ComputeRegionNetworkFirewallPoliciesGetRuleRequest object.

  Fields:
    firewallPolicy: Name of the firewall policy to which the queried rule
      belongs.
    priority: The priority of the rule to get from the firewall policy.
    project: Project ID for this request.
    region: Name of the region scoping this request.
  """
    firewallPolicy = _messages.StringField(1, required=True)
    priority = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    project = _messages.StringField(3, required=True)
    region = _messages.StringField(4, required=True)