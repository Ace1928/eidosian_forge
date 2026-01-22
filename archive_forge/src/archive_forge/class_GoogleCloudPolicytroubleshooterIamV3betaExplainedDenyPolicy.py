from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterIamV3betaExplainedDenyPolicy(_messages.Message):
    """Details about how a specific IAM deny policy Policy contributed to the
  access check.

  Enums:
    DenyAccessStateValueValuesEnum: Required. Indicates whether _this policy_
      denies the specified permission to the specified principal for the
      specified resource. This field does _not_ indicate whether the principal
      actually has the permission for the resource. There might be another
      policy that overrides this policy. To determine whether the principal
      actually has the permission, use the `overall_access_state` field in the
      TroubleshootIamPolicyResponse.
    RelevanceValueValuesEnum: The relevance of this policy to the overall
      access state in the TroubleshootIamPolicyResponse. If the sender of the
      request does not have access to the policy, this field is omitted.

  Fields:
    denyAccessState: Required. Indicates whether _this policy_ denies the
      specified permission to the specified principal for the specified
      resource. This field does _not_ indicate whether the principal actually
      has the permission for the resource. There might be another policy that
      overrides this policy. To determine whether the principal actually has
      the permission, use the `overall_access_state` field in the
      TroubleshootIamPolicyResponse.
    policy: The IAM deny policy attached to the resource. If the sender of the
      request does not have access to the policy, this field is omitted.
    relevance: The relevance of this policy to the overall access state in the
      TroubleshootIamPolicyResponse. If the sender of the request does not
      have access to the policy, this field is omitted.
    ruleExplanations: Details about how each rule in the policy affects the
      principal's inability to use the permission for the resource. The order
      of the deny rule matches the order of the rules in the deny policy. If
      the sender of the request does not have access to the policy, this field
      is omitted.
  """

    class DenyAccessStateValueValuesEnum(_messages.Enum):
        """Required. Indicates whether _this policy_ denies the specified
    permission to the specified principal for the specified resource. This
    field does _not_ indicate whether the principal actually has the
    permission for the resource. There might be another policy that overrides
    this policy. To determine whether the principal actually has the
    permission, use the `overall_access_state` field in the
    TroubleshootIamPolicyResponse.

    Values:
      DENY_ACCESS_STATE_UNSPECIFIED: Not specified.
      DENY_ACCESS_STATE_DENIED: The deny policy denies the principal the
        permission.
      DENY_ACCESS_STATE_NOT_DENIED: The deny policy doesn't deny the principal
        the permission.
      DENY_ACCESS_STATE_UNKNOWN_CONDITIONAL: The deny policy denies the
        principal the permission if a condition expression evaluates to
        `true`. However, the sender of the request didn't provide enough
        context for Policy Troubleshooter to evaluate the condition
        expression.
      DENY_ACCESS_STATE_UNKNOWN_INFO: The sender of the request does not have
        access to all of the deny policies that Policy Troubleshooter needs to
        evaluate the principal's access.
    """
        DENY_ACCESS_STATE_UNSPECIFIED = 0
        DENY_ACCESS_STATE_DENIED = 1
        DENY_ACCESS_STATE_NOT_DENIED = 2
        DENY_ACCESS_STATE_UNKNOWN_CONDITIONAL = 3
        DENY_ACCESS_STATE_UNKNOWN_INFO = 4

    class RelevanceValueValuesEnum(_messages.Enum):
        """The relevance of this policy to the overall access state in the
    TroubleshootIamPolicyResponse. If the sender of the request does not have
    access to the policy, this field is omitted.

    Values:
      HEURISTIC_RELEVANCE_UNSPECIFIED: Not specified.
      HEURISTIC_RELEVANCE_NORMAL: The data point has a limited effect on the
        result. Changing the data point is unlikely to affect the overall
        determination.
      HEURISTIC_RELEVANCE_HIGH: The data point has a strong effect on the
        result. Changing the data point is likely to affect the overall
        determination.
    """
        HEURISTIC_RELEVANCE_UNSPECIFIED = 0
        HEURISTIC_RELEVANCE_NORMAL = 1
        HEURISTIC_RELEVANCE_HIGH = 2
    denyAccessState = _messages.EnumField('DenyAccessStateValueValuesEnum', 1)
    policy = _messages.MessageField('GoogleIamV2Policy', 2)
    relevance = _messages.EnumField('RelevanceValueValuesEnum', 3)
    ruleExplanations = _messages.MessageField('GoogleCloudPolicytroubleshooterIamV3betaDenyRuleExplanation', 4, repeated=True)