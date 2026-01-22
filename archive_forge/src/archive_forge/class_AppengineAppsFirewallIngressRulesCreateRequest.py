from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppengineAppsFirewallIngressRulesCreateRequest(_messages.Message):
    """A AppengineAppsFirewallIngressRulesCreateRequest object.

  Fields:
    firewallRule: A FirewallRule resource to be passed as the request body.
    parent: Name of the parent Firewall collection in which to create a new
      rule. Example: apps/myapp/firewall/ingressRules.
  """
    firewallRule = _messages.MessageField('FirewallRule', 1)
    parent = _messages.StringField(2, required=True)