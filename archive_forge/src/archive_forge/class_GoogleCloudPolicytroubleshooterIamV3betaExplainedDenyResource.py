from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterIamV3betaExplainedDenyResource(_messages.Message):
    """Details about how a specific resource contributed to the deny policy
  evaluation.

  Enums:
    DenyAccessStateValueValuesEnum: Required. Indicates whether any policies
      attached to _this resource_ deny the specific permission to the
      specified principal for the specified resource. This field does _not_
      indicate whether the principal actually has the permission for the
      resource. There might be another policy that overrides this policy. To
      determine whether the principal actually has the permission, use the
      `overall_access_state` field in the TroubleshootIamPolicyResponse.
    RelevanceValueValuesEnum: The relevance of this policy to the overall
      access state in the TroubleshootIamPolicyResponse. If the sender of the
      request does not have access to the policy, this field is omitted.

  Fields:
    denyAccessState: Required. Indicates whether any policies attached to
      _this resource_ deny the specific permission to the specified principal
      for the specified resource. This field does _not_ indicate whether the
      principal actually has the permission for the resource. There might be
      another policy that overrides this policy. To determine whether the
      principal actually has the permission, use the `overall_access_state`
      field in the TroubleshootIamPolicyResponse.
    explainedPolicies: List of IAM deny policies that were evaluated to check
      the principal's denied permissions, with annotations to indicate how
      each policy contributed to the final result.
    fullResourceName: The full resource name that identifies the resource. For
      example, `//compute.googleapis.com/projects/my-project/zones/us-
      central1-a/instances/my-instance`. If the sender of the request does not
      have access to the policy, this field is omitted. For examples of full
      resource names for Google Cloud services, see
      https://cloud.google.com/iam/help/troubleshooter/full-resource-names.
    relevance: The relevance of this policy to the overall access state in the
      TroubleshootIamPolicyResponse. If the sender of the request does not
      have access to the policy, this field is omitted.
  """

    class DenyAccessStateValueValuesEnum(_messages.Enum):
        """Required. Indicates whether any policies attached to _this resource_
    deny the specific permission to the specified principal for the specified
    resource. This field does _not_ indicate whether the principal actually
    has the permission for the resource. There might be another policy that
    overrides this policy. To determine whether the principal actually has the
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
    explainedPolicies = _messages.MessageField('GoogleCloudPolicytroubleshooterIamV3betaExplainedDenyPolicy', 2, repeated=True)
    fullResourceName = _messages.StringField(3)
    relevance = _messages.EnumField('RelevanceValueValuesEnum', 4)