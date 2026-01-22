from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeSecurityPoliciesRemoveRuleRequest(_messages.Message):
    """A ComputeSecurityPoliciesRemoveRuleRequest object.

  Fields:
    priority: The priority of the rule to remove from the security policy.
    project: Project ID for this request.
    securityPolicy: Name of the security policy to update.
  """
    priority = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    project = _messages.StringField(2, required=True)
    securityPolicy = _messages.StringField(3, required=True)