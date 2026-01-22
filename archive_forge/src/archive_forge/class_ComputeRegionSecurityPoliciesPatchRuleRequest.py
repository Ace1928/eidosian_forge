from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeRegionSecurityPoliciesPatchRuleRequest(_messages.Message):
    """A ComputeRegionSecurityPoliciesPatchRuleRequest object.

  Fields:
    priority: The priority of the rule to patch.
    project: Project ID for this request.
    region: Name of the region scoping this request.
    securityPolicy: Name of the security policy to update.
    securityPolicyRule: A SecurityPolicyRule resource to be passed as the
      request body.
    updateMask: Indicates fields to be cleared as part of this request.
    validateOnly: If true, the request will not be committed.
  """
    priority = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    project = _messages.StringField(2, required=True)
    region = _messages.StringField(3, required=True)
    securityPolicy = _messages.StringField(4, required=True)
    securityPolicyRule = _messages.MessageField('SecurityPolicyRule', 5)
    updateMask = _messages.StringField(6)
    validateOnly = _messages.BooleanField(7)