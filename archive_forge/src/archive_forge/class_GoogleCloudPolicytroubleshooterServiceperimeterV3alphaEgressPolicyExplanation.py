from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterServiceperimeterV3alphaEgressPolicyExplanation(_messages.Message):
    """Explanation of an egress policy NextTAG: 8

  Enums:
    ApiOperationEvalStatesValueListEntryValuesEnum:
    EgressPolicyEvalStateValueValuesEnum: The overall evaluation state of the
      egress policy
    ExternalResourceEvalStatesValueListEntryValuesEnum:
    IdentityTypeEvalStateValueValuesEnum: Details of the evaluation state of
      the identity type
    ResourceEvalStatesValueListEntryValuesEnum:

  Fields:
    apiOperationEvalStates: Details of the evaluation states of api operations
    egressPolicyEvalState: The overall evaluation state of the egress policy
    externalResourceEvalStates: Details of the evaluation states of external
      resources
    identityExplanations: Detailed explanation of the identities.
    identityTypeEvalState: Details of the evaluation state of the identity
      type
    resourceEvalStates: Details of the evaluation states of resources
  """

    class ApiOperationEvalStatesValueListEntryValuesEnum(_messages.Enum):
        """ApiOperationEvalStatesValueListEntryValuesEnum enum type.

    Values:
      API_OPERATION_EVAL_STATE_UNSPECIFIED: Not used
      API_OPERATION_EVAL_STATE_MATCH: The request matches the api operation
      API_OPERATION_EVAL_STATE_NOT_MATCH: The request doesn't match the api
        operation
    """
        API_OPERATION_EVAL_STATE_UNSPECIFIED = 0
        API_OPERATION_EVAL_STATE_MATCH = 1
        API_OPERATION_EVAL_STATE_NOT_MATCH = 2

    class EgressPolicyEvalStateValueValuesEnum(_messages.Enum):
        """The overall evaluation state of the egress policy

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

    class ExternalResourceEvalStatesValueListEntryValuesEnum(_messages.Enum):
        """ExternalResourceEvalStatesValueListEntryValuesEnum enum type.

    Values:
      RESOURCE_EVAL_STATE_UNSPECIFIED: Not used
      RESOURCE_EVAL_STATE_MATCH: The request matches the resource
      RESOURCE_EVAL_STATE_NOT_MATCH: The request doesn't match the resource
    """
        RESOURCE_EVAL_STATE_UNSPECIFIED = 0
        RESOURCE_EVAL_STATE_MATCH = 1
        RESOURCE_EVAL_STATE_NOT_MATCH = 2

    class IdentityTypeEvalStateValueValuesEnum(_messages.Enum):
        """Details of the evaluation state of the identity type

    Values:
      IDENTITY_TYPE_EVAL_STATE_UNSPECIFIED: Not used
      IDENTITY_TYPE_EVAL_STATE_GRANTED: The request type matches the identity
      IDENTITY_TYPE_EVAL_STATE_NOT_GRANTED: The request type doesn't match the
        identity
      IDENTITY_TYPE_EVAL_STATE_NOT_SUPPORTED: The identity type is not
        supported
    """
        IDENTITY_TYPE_EVAL_STATE_UNSPECIFIED = 0
        IDENTITY_TYPE_EVAL_STATE_GRANTED = 1
        IDENTITY_TYPE_EVAL_STATE_NOT_GRANTED = 2
        IDENTITY_TYPE_EVAL_STATE_NOT_SUPPORTED = 3

    class ResourceEvalStatesValueListEntryValuesEnum(_messages.Enum):
        """ResourceEvalStatesValueListEntryValuesEnum enum type.

    Values:
      RESOURCE_EVAL_STATE_UNSPECIFIED: Not used
      RESOURCE_EVAL_STATE_MATCH: The request matches the resource
      RESOURCE_EVAL_STATE_NOT_MATCH: The request doesn't match the resource
    """
        RESOURCE_EVAL_STATE_UNSPECIFIED = 0
        RESOURCE_EVAL_STATE_MATCH = 1
        RESOURCE_EVAL_STATE_NOT_MATCH = 2
    apiOperationEvalStates = _messages.EnumField('ApiOperationEvalStatesValueListEntryValuesEnum', 1, repeated=True)
    egressPolicyEvalState = _messages.EnumField('EgressPolicyEvalStateValueValuesEnum', 2)
    externalResourceEvalStates = _messages.EnumField('ExternalResourceEvalStatesValueListEntryValuesEnum', 3, repeated=True)
    identityExplanations = _messages.MessageField('GoogleCloudPolicytroubleshooterServiceperimeterV3alphaIdentityExplanation', 4, repeated=True)
    identityTypeEvalState = _messages.EnumField('IdentityTypeEvalStateValueValuesEnum', 5)
    resourceEvalStates = _messages.EnumField('ResourceEvalStatesValueListEntryValuesEnum', 6, repeated=True)