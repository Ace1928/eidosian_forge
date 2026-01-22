from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksecurityProjectsLocationsGatewaySecurityPoliciesRulesCreateRequest(_messages.Message):
    """A
  NetworksecurityProjectsLocationsGatewaySecurityPoliciesRulesCreateRequest
  object.

  Fields:
    gatewaySecurityPolicyRule: A GatewaySecurityPolicyRule resource to be
      passed as the request body.
    gatewaySecurityPolicyRuleId: The ID to use for the rule, which will become
      the final component of the rule's resource name. This value should be
      4-63 characters, and valid characters are /a-z-/.
    parent: Required. The parent where this rule will be created. Format :
      projects/{project}/location/{location}/gatewaySecurityPolicies/*
  """
    gatewaySecurityPolicyRule = _messages.MessageField('GatewaySecurityPolicyRule', 1)
    gatewaySecurityPolicyRuleId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)