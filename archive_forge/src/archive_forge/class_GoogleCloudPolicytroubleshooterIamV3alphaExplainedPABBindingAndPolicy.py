from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterIamV3alphaExplainedPABBindingAndPolicy(_messages.Message):
    """Details about how a Principal Access Boundary binding and policy
  contributes to the Principal Access Boundary explanation, with annotations
  to indicate how the binding and policy contribute to the overall access
  state.

  Enums:
    BindingAndPolicyAccessStateValueValuesEnum: Output only. Indicates whether
      the principal is allowed to access the specified resource based on
      evaluating the binding and policy.
    RelevanceValueValuesEnum: The relevance of this Principal Access Boundary
      binding and policy to the overall access state.

  Fields:
    bindingAndPolicyAccessState: Output only. Indicates whether the principal
      is allowed to access the specified resource based on evaluating the
      binding and policy.
    explainedPolicy: Optional. Details about how this policy contributes to
      the Principal Access Boundary explanation, with annotations to indicate
      how the policy contributes to the overall access state. If the caller
      doesn't have permission to view the policy in the binding, this field is
      omitted.
    explainedPolicyBinding: Details about how this binding contributes to the
      Principal Access Boundary explanation, with annotations to indicate how
      the binding contributes to the overall access state.
    relevance: The relevance of this Principal Access Boundary binding and
      policy to the overall access state.
  """

    class BindingAndPolicyAccessStateValueValuesEnum(_messages.Enum):
        """Output only. Indicates whether the principal is allowed to access the
    specified resource based on evaluating the binding and policy.

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
        """The relevance of this Principal Access Boundary binding and policy to
    the overall access state.

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
    bindingAndPolicyAccessState = _messages.EnumField('BindingAndPolicyAccessStateValueValuesEnum', 1)
    explainedPolicy = _messages.MessageField('GoogleCloudPolicytroubleshooterIamV3alphaExplainedPABPolicy', 2)
    explainedPolicyBinding = _messages.MessageField('GoogleCloudPolicytroubleshooterIamV3alphaExplainedPolicyBinding', 3)
    relevance = _messages.EnumField('RelevanceValueValuesEnum', 4)