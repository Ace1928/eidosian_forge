from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ForwardingRuleTarget(_messages.Message):
    """Definition of FIT Target which describes a specific forwarding rule.

  Fields:
    forwardingRule: Reference to the targeted ForwardingRule (URI) See more:
      https://cloud.google.com/compute/docs/reference/rest/v1/forwardingRules
  """
    forwardingRule = _messages.StringField(1)