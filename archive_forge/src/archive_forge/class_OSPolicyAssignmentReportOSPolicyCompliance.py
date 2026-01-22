from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OSPolicyAssignmentReportOSPolicyCompliance(_messages.Message):
    """Compliance data for an OS policy

  Enums:
    ComplianceStateValueValuesEnum: The compliance state of the OS policy.

  Fields:
    complianceState: The compliance state of the OS policy.
    complianceStateReason: The reason for the OS policy to be in an unknown
      compliance state. This field is always populated when `compliance_state`
      is `UNKNOWN`. If populated, the field can contain one of the following
      values: * `vm-not-running`: The VM was not running. * `os-policies-not-
      supported-by-agent`: The version of the OS Config agent running on the
      VM does not support running OS policies. * `no-agent-detected`: The OS
      Config agent is not detected for the VM. * `resource-execution-errors`:
      The OS Config agent encountered errors while executing one or more
      resources in the policy. See `os_policy_resource_compliances` for
      details. * `task-timeout`: The task sent to the agent to apply the
      policy timed out. * `unexpected-agent-state`: The OS Config agent did
      not report the final status of the task that attempted to apply the
      policy. Instead, the agent unexpectedly started working on a different
      task. This mostly happens when the agent or VM unexpectedly restarts
      while applying OS policies. * `internal-service-errors`: Internal
      service errors were encountered while attempting to apply the policy.
    osPolicyId: The OS policy id
    osPolicyResourceCompliances: Compliance data for each resource within the
      policy that is applied to the VM.
  """

    class ComplianceStateValueValuesEnum(_messages.Enum):
        """The compliance state of the OS policy.

    Values:
      UNKNOWN: The policy is in an unknown compliance state. Refer to the
        field `compliance_state_reason` to learn the exact reason for the
        policy to be in this compliance state.
      COMPLIANT: Policy is compliant. The policy is compliant if all the
        underlying resources are also compliant.
      NON_COMPLIANT: Policy is non-compliant. The policy is non-compliant if
        one or more underlying resources are non-compliant.
    """
        UNKNOWN = 0
        COMPLIANT = 1
        NON_COMPLIANT = 2
    complianceState = _messages.EnumField('ComplianceStateValueValuesEnum', 1)
    complianceStateReason = _messages.StringField(2)
    osPolicyId = _messages.StringField(3)
    osPolicyResourceCompliances = _messages.MessageField('OSPolicyAssignmentReportOSPolicyComplianceOSPolicyResourceCompliance', 4, repeated=True)