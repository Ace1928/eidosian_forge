from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterServiceperimeterV3alphaEgressPoliciesExplanation(_messages.Message):
    """Explanation of egress policies NextTAG: 5

  Enums:
    EgressPolicyEvalStatesValueListEntryValuesEnum:

  Fields:
    egressPolicyEvalStates: Details about the evaluation state of the egress
      policy
    egressPolicyExplanations: Explanations of egress policies
    sourceResource: The source resource to egress from
    targetResource: The target resource to egress to
  """

    class EgressPolicyEvalStatesValueListEntryValuesEnum(_messages.Enum):
        """EgressPolicyEvalStatesValueListEntryValuesEnum enum type.

    Values:
      EGRESS_POLICY_EVAL_STATE_UNSPECIFIED: Not used
      EGRESS_POLICY_EVAL_STATE_IN_SAME_SERVICE_PERIMETER: The resources are in
        the same regular service perimeter
      EGRESS_POLICY_EVAL_STATE_GRANTED_OVER_BRIDGE: The resources are in the
        same bridge service perimeter
      EGRESS_POLICY_EVAL_STATE_GRANTED_BY_POLICY: The request is granted by
        the egress policy
      EGRESS_POLICY_EVAL_STATE_DENIED_BY_POLICY: The request is denied by the
        egress policy
      EGRESS_POLICY_EVAL_STATE_NOT_APPLICABLE: The egress policy is applicable
        for the request
    """
        EGRESS_POLICY_EVAL_STATE_UNSPECIFIED = 0
        EGRESS_POLICY_EVAL_STATE_IN_SAME_SERVICE_PERIMETER = 1
        EGRESS_POLICY_EVAL_STATE_GRANTED_OVER_BRIDGE = 2
        EGRESS_POLICY_EVAL_STATE_GRANTED_BY_POLICY = 3
        EGRESS_POLICY_EVAL_STATE_DENIED_BY_POLICY = 4
        EGRESS_POLICY_EVAL_STATE_NOT_APPLICABLE = 5
    egressPolicyEvalStates = _messages.EnumField('EgressPolicyEvalStatesValueListEntryValuesEnum', 1, repeated=True)
    egressPolicyExplanations = _messages.MessageField('GoogleCloudPolicytroubleshooterServiceperimeterV3alphaEgressPolicyExplanation', 2, repeated=True)
    sourceResource = _messages.MessageField('GoogleCloudPolicytroubleshooterServiceperimeterV3alphaResource', 3)
    targetResource = _messages.MessageField('GoogleCloudPolicytroubleshooterServiceperimeterV3alphaResource', 4)