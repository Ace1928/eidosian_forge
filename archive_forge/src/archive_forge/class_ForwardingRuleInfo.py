from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ForwardingRuleInfo(_messages.Message):
    """For display only. Metadata associated with a Compute Engine forwarding
  rule.

  Fields:
    displayName: Name of a Compute Engine forwarding rule.
    matchedPortRange: Port range defined in the forwarding rule that matches
      the test.
    matchedProtocol: Protocol defined in the forwarding rule that matches the
      test.
    networkUri: Network URI. Only valid for Internal Load Balancer.
    target: Target type of the forwarding rule.
    uri: URI of a Compute Engine forwarding rule.
    vip: VIP of the forwarding rule.
  """
    displayName = _messages.StringField(1)
    matchedPortRange = _messages.StringField(2)
    matchedProtocol = _messages.StringField(3)
    networkUri = _messages.StringField(4)
    target = _messages.StringField(5)
    uri = _messages.StringField(6)
    vip = _messages.StringField(7)