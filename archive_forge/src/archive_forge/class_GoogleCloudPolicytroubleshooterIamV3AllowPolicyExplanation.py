from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterIamV3AllowPolicyExplanation(_messages.Message):
    """Details about how the relevant IAM allow policies affect the final
  access state.

  Enums:
    AllowAccessStateValueValuesEnum: Indicates whether the principal has the
      specified permission for the specified resource, based on evaluating all
      applicable IAM allow policies.
    RelevanceValueValuesEnum: The relevance of the allow policy type to the
      overall access state.

  Fields:
    allowAccessState: Indicates whether the principal has the specified
      permission for the specified resource, based on evaluating all
      applicable IAM allow policies.
    explainedPolicies: List of IAM allow policies that were evaluated to check
      the principal's permissions, with annotations to indicate how each
      policy contributed to the final result. The list of policies includes
      the policy for the resource itself, as well as allow policies that are
      inherited from higher levels of the resource hierarchy, including the
      organization, the folder, and the project. To learn more about the
      resource hierarchy, see https://cloud.google.com/iam/help/resource-
      hierarchy.
    relevance: The relevance of the allow policy type to the overall access
      state.
  """

    class AllowAccessStateValueValuesEnum(_messages.Enum):
        """Indicates whether the principal has the specified permission for the
    specified resource, based on evaluating all applicable IAM allow policies.

    Values:
      ALLOW_ACCESS_STATE_UNSPECIFIED: Not specified.
      ALLOW_ACCESS_STATE_GRANTED: The allow policy gives the principal the
        permission.
      ALLOW_ACCESS_STATE_NOT_GRANTED: The allow policy doesn't give the
        principal the permission.
      ALLOW_ACCESS_STATE_UNKNOWN_CONDITIONAL: The allow policy gives the
        principal the permission if a condition expression evaluate to `true`.
        However, the sender of the request didn't provide enough context for
        Policy Troubleshooter to evaluate the condition expression.
      ALLOW_ACCESS_STATE_UNKNOWN_INFO: The sender of the request doesn't have
        access to all of the allow policies that Policy Troubleshooter needs
        to evaluate the principal's access.
    """
        ALLOW_ACCESS_STATE_UNSPECIFIED = 0
        ALLOW_ACCESS_STATE_GRANTED = 1
        ALLOW_ACCESS_STATE_NOT_GRANTED = 2
        ALLOW_ACCESS_STATE_UNKNOWN_CONDITIONAL = 3
        ALLOW_ACCESS_STATE_UNKNOWN_INFO = 4

    class RelevanceValueValuesEnum(_messages.Enum):
        """The relevance of the allow policy type to the overall access state.

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
    allowAccessState = _messages.EnumField('AllowAccessStateValueValuesEnum', 1)
    explainedPolicies = _messages.MessageField('GoogleCloudPolicytroubleshooterIamV3ExplainedAllowPolicy', 2, repeated=True)
    relevance = _messages.EnumField('RelevanceValueValuesEnum', 3)