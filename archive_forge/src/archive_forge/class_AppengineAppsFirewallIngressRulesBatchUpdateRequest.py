from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppengineAppsFirewallIngressRulesBatchUpdateRequest(_messages.Message):
    """A AppengineAppsFirewallIngressRulesBatchUpdateRequest object.

  Fields:
    batchUpdateIngressRulesRequest: A BatchUpdateIngressRulesRequest resource
      to be passed as the request body.
    name: Name of the Firewall collection to set. Example:
      apps/myapp/firewall/ingressRules.
  """
    batchUpdateIngressRulesRequest = _messages.MessageField('BatchUpdateIngressRulesRequest', 1)
    name = _messages.StringField(2, required=True)