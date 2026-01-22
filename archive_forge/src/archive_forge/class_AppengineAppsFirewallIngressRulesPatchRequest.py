from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppengineAppsFirewallIngressRulesPatchRequest(_messages.Message):
    """A AppengineAppsFirewallIngressRulesPatchRequest object.

  Fields:
    firewallRule: A FirewallRule resource to be passed as the request body.
    name: Name of the Firewall resource to update. Example:
      apps/myapp/firewall/ingressRules/100.
    updateMask: Standard field mask for the set of fields to be updated.
  """
    firewallRule = _messages.MessageField('FirewallRule', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)