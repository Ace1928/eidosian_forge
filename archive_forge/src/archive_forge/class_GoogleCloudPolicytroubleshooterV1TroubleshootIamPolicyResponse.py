from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterV1TroubleshootIamPolicyResponse(_messages.Message):
    """Response for TroubleshootIamPolicy.

  Enums:
    AccessValueValuesEnum: Indicates whether the principal has the specified
      permission for the specified resource, based on evaluating all of the
      applicable IAM policies.

  Fields:
    access: Indicates whether the principal has the specified permission for
      the specified resource, based on evaluating all of the applicable IAM
      policies.
    errors: The general errors contained in the troubleshooting response.
    explainedPolicies: List of IAM policies that were evaluated to check the
      principal's permissions, with annotations to indicate how each policy
      contributed to the final result. The list of policies can include the
      policy for the resource itself. It can also include policies that are
      inherited from higher levels of the resource hierarchy, including the
      organization, the folder, and the project. To learn more about the
      resource hierarchy, see https://cloud.google.com/iam/help/resource-
      hierarchy.
  """

    class AccessValueValuesEnum(_messages.Enum):
        """Indicates whether the principal has the specified permission for the
    specified resource, based on evaluating all of the applicable IAM
    policies.

    Values:
      ACCESS_STATE_UNSPECIFIED: Default value. This value is unused.
      GRANTED: The principal has the permission.
      NOT_GRANTED: The principal does not have the permission.
      UNKNOWN_CONDITIONAL: The principal has the permission only if a
        condition expression evaluates to `true`.
      UNKNOWN_INFO_DENIED: The sender of the request does not have access to
        all of the policies that Policy Troubleshooter needs to evaluate.
    """
        ACCESS_STATE_UNSPECIFIED = 0
        GRANTED = 1
        NOT_GRANTED = 2
        UNKNOWN_CONDITIONAL = 3
        UNKNOWN_INFO_DENIED = 4
    access = _messages.EnumField('AccessValueValuesEnum', 1)
    errors = _messages.MessageField('GoogleRpcStatus', 2, repeated=True)
    explainedPolicies = _messages.MessageField('GoogleCloudPolicytroubleshooterV1ExplainedPolicy', 3, repeated=True)