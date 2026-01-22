from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ControlDetails(_messages.Message):
    """Evaluation details for a control

  Enums:
    ComplianceStateValueValuesEnum: Output only. Overall status of the
      findings for the control.

  Fields:
    complianceState: Output only. Overall status of the findings for the
      control.
    control: The control for which the findings are being reported.
  """

    class ComplianceStateValueValuesEnum(_messages.Enum):
        """Output only. Overall status of the findings for the control.

    Values:
      COMPLIANCE_STATE_UNSPECIFIED: Unspecified. Invalid state.
      COMPLIANT: Compliant.
      VIOLATION: Violation.
      UNKNOWN: Unknown, requires manual review
      ERROR: Error while computing status.
    """
        COMPLIANCE_STATE_UNSPECIFIED = 0
        COMPLIANT = 1
        VIOLATION = 2
        UNKNOWN = 3
        ERROR = 4
    complianceState = _messages.EnumField('ComplianceStateValueValuesEnum', 1)
    control = _messages.MessageField('Control', 2)