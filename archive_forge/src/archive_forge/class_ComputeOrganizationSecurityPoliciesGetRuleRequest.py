from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeOrganizationSecurityPoliciesGetRuleRequest(_messages.Message):
    """A ComputeOrganizationSecurityPoliciesGetRuleRequest object.

  Fields:
    priority: The priority of the rule to get from the security policy.
    securityPolicy: Name of the security policy to which the queried rule
      belongs.
  """
    priority = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    securityPolicy = _messages.StringField(2, required=True)