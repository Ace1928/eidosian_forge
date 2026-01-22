from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceOSPoliciesCompliance(_messages.Message):
    """This API resource represents the OS policies compliance data for a
  Compute Engine virtual machine (VM) instance at a given point in time. A
  Compute Engine VM can have multiple OS policy assignments, and each
  assignment can have multiple OS policies. As a result, multiple OS policies
  could be applied to a single VM. You can use this API resource to determine
  both the compliance state of your VM as well as the compliance state of an
  individual OS policy. For more information, see [View
  compliance](https://cloud.google.com/compute/docs/os-configuration-
  management/view-compliance).

  Enums:
    StateValueValuesEnum: Output only. Compliance state of the VM.

  Fields:
    detailedState: Output only. Detailed compliance state of the VM. This
      field is populated only when compliance state is `UNKNOWN`. It may
      contain one of the following values: * `no-compliance-data`: Compliance
      data is not available for this VM. * `no-agent-detected`: OS Config
      agent is not detected for this VM. * `config-not-supported-by-agent`:
      The version of the OS Config agent running on this VM does not support
      configuration management. * `inactive`: VM is not running. * `internal-
      service-errors`: There were internal service errors encountered while
      enforcing compliance. * `agent-errors`: OS config agent encountered
      errors while enforcing compliance.
    detailedStateReason: Output only. The reason for the `detailed_state` of
      the VM (if any).
    instance: Output only. The Compute Engine VM instance name.
    lastComplianceCheckTime: Output only. Timestamp of the last compliance
      check for the VM.
    lastComplianceRunId: Output only. Unique identifier for the last
      compliance run. This id will be logged by the OS config agent during a
      compliance run and can be used for debugging and tracing purpose.
    name: Output only. The `InstanceOSPoliciesCompliance` API resource name.
      Format: `projects/{project_number}/locations/{location}/instanceOSPolici
      esCompliances/{instance_id}`
    osPolicyCompliances: Output only. Compliance data for each `OSPolicy` that
      is applied to the VM.
    state: Output only. Compliance state of the VM.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. Compliance state of the VM.

    Values:
      OS_POLICY_COMPLIANCE_STATE_UNSPECIFIED: Default value. This value is
        unused.
      COMPLIANT: Compliant state.
      NON_COMPLIANT: Non-compliant state
      UNKNOWN: Unknown compliance state.
      NO_OS_POLICIES_APPLICABLE: No applicable OS policies were found for the
        instance. This state is only applicable to the instance.
    """
        OS_POLICY_COMPLIANCE_STATE_UNSPECIFIED = 0
        COMPLIANT = 1
        NON_COMPLIANT = 2
        UNKNOWN = 3
        NO_OS_POLICIES_APPLICABLE = 4
    detailedState = _messages.StringField(1)
    detailedStateReason = _messages.StringField(2)
    instance = _messages.StringField(3)
    lastComplianceCheckTime = _messages.StringField(4)
    lastComplianceRunId = _messages.StringField(5)
    name = _messages.StringField(6)
    osPolicyCompliances = _messages.MessageField('InstanceOSPoliciesComplianceOSPolicyCompliance', 7, repeated=True)
    state = _messages.EnumField('StateValueValuesEnum', 8)