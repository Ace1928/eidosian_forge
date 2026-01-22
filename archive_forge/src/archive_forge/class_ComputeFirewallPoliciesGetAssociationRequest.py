from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeFirewallPoliciesGetAssociationRequest(_messages.Message):
    """A ComputeFirewallPoliciesGetAssociationRequest object.

  Fields:
    firewallPolicy: Name of the firewall policy to which the queried rule
      belongs.
    name: The name of the association to get from the firewall policy.
  """
    firewallPolicy = _messages.StringField(1, required=True)
    name = _messages.StringField(2)