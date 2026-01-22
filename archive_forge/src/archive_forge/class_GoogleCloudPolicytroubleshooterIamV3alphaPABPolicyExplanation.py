from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterIamV3alphaPABPolicyExplanation(_messages.Message):
    """Details about how the relevant Principal Access Boundary policies affect
  the overall access state.

  Enums:
    PrincipalAccessBoundaryAccessStateValueValuesEnum: Output only. Indicates
      whether the principal is allowed to access specified resource, based on
      evaluating all applicable Principal Access Boundary bindings and
      policies.
    RelevanceValueValuesEnum: The relevance of the Principal Access Boundary
      access state to the overall access state.

  Fields:
    explainedBindingsAndPolicies: List of Principal Access Boundary policies
      and bindings that are applicable to the principal's access state, with
      annotations to indicate how each binding and policy contributes to the
      overall access state.
    principalAccessBoundaryAccessState: Output only. Indicates whether the
      principal is allowed to access specified resource, based on evaluating
      all applicable Principal Access Boundary bindings and policies.
    relevance: The relevance of the Principal Access Boundary access state to
      the overall access state.
  """

    class PrincipalAccessBoundaryAccessStateValueValuesEnum(_messages.Enum):
        """Output only. Indicates whether the principal is allowed to access
    specified resource, based on evaluating all applicable Principal Access
    Boundary bindings and policies.

    Values:
      PAB_ACCESS_STATE_UNSPECIFIED: Not specified.
      PAB_ACCESS_STATE_ALLOWED: The PAB component allows the principal's
        access to the specified resource.
      PAB_ACCESS_STATE_NOT_ALLOWED: The PAB component doesn't allow the
        principal's access to the specified resource.
      PAB_ACCESS_STATE_NOT_ENFORCED: The PAB component is not enforced on the
        principal, or the specified resource. This state refers to 2 specific
        scenarios: - The service that the specified resource belongs to is not
        enforced by PAB at the policy version. - The binding doesn't apply to
        the principal, hence the policy is not enforced as a result.
      PAB_ACCESS_STATE_UNKNOWN_INFO: The sender of the request does not have
        access to the PAB component, or the relevant data to explain the PAB
        component.
    """
        PAB_ACCESS_STATE_UNSPECIFIED = 0
        PAB_ACCESS_STATE_ALLOWED = 1
        PAB_ACCESS_STATE_NOT_ALLOWED = 2
        PAB_ACCESS_STATE_NOT_ENFORCED = 3
        PAB_ACCESS_STATE_UNKNOWN_INFO = 4

    class RelevanceValueValuesEnum(_messages.Enum):
        """The relevance of the Principal Access Boundary access state to the
    overall access state.

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
    explainedBindingsAndPolicies = _messages.MessageField('GoogleCloudPolicytroubleshooterIamV3alphaExplainedPABBindingAndPolicy', 1, repeated=True)
    principalAccessBoundaryAccessState = _messages.EnumField('PrincipalAccessBoundaryAccessStateValueValuesEnum', 2)
    relevance = _messages.EnumField('RelevanceValueValuesEnum', 3)