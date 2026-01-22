from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OSPolicyResourceCompliance(_messages.Message):
    """Compliance data for an OS policy resource.

  Enums:
    StateValueValuesEnum: Compliance state of the OS policy resource.

  Fields:
    configSteps: Ordered list of configuration steps taken by the agent for
      the OS policy resource.
    execResourceOutput: ExecResource specific output.
    osPolicyResourceId: The id of the OS policy resource.
    state: Compliance state of the OS policy resource.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Compliance state of the OS policy resource.

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
    configSteps = _messages.MessageField('OSPolicyResourceConfigStep', 1, repeated=True)
    execResourceOutput = _messages.MessageField('OSPolicyResourceComplianceExecResourceOutput', 2)
    osPolicyResourceId = _messages.StringField(3)
    state = _messages.EnumField('StateValueValuesEnum', 4)