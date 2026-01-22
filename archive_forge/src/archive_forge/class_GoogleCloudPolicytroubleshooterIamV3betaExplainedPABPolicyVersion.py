from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterIamV3betaExplainedPABPolicyVersion(_messages.Message):
    """Details about how a Principal Access Boundary policy's version
  contributes to the policy's enforcement state.

  Enums:
    EnforcementStateValueValuesEnum: Output only. Indicates whether the policy
      is enforced based on its version.

  Fields:
    enforcementState: Output only. Indicates whether the policy is enforced
      based on its version.
    version: Output only. The actual version of the policy. - If the policy
      uses static version, this field is the chosen static version. - If the
      policy uses dynamic version, this field is the effective latest version.
  """

    class EnforcementStateValueValuesEnum(_messages.Enum):
        """Output only. Indicates whether the policy is enforced based on its
    version.

    Values:
      PAB_POLICY_ENFORCEMENT_STATE_UNSPECIFIED: An error occurred when
        checking whether a Principal Access Boundary policy is enforced based
        on its version.
      PAB_POLICY_ENFORCEMENT_STATE_ENFORCED: The Principal Access Boundary
        policy is enforced based on its version.
      PAB_POLICY_ENFORCEMENT_STATE_NOT_ENFORCED: The Principal Access Boundary
        policy is not enforced based on its version.
    """
        PAB_POLICY_ENFORCEMENT_STATE_UNSPECIFIED = 0
        PAB_POLICY_ENFORCEMENT_STATE_ENFORCED = 1
        PAB_POLICY_ENFORCEMENT_STATE_NOT_ENFORCED = 2
    enforcementState = _messages.EnumField('EnforcementStateValueValuesEnum', 1)
    version = _messages.IntegerField(2, variant=_messages.Variant.INT32)