from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CompliancePostureConfig(_messages.Message):
    """CompliancePostureConfig defines the settings needed to enable/disable
  features for the Compliance Posture.

  Enums:
    ModeValueValuesEnum: Defines the enablement mode for Compliance Posture.

  Fields:
    complianceStandards: List of enabled compliance standards.
    mode: Defines the enablement mode for Compliance Posture.
  """

    class ModeValueValuesEnum(_messages.Enum):
        """Defines the enablement mode for Compliance Posture.

    Values:
      MODE_UNSPECIFIED: Default value not specified.
      DISABLED: Disables Compliance Posture features on the cluster.
      ENABLED: Enables Compliance Posture features on the cluster.
    """
        MODE_UNSPECIFIED = 0
        DISABLED = 1
        ENABLED = 2
    complianceStandards = _messages.MessageField('ComplianceStandard', 1, repeated=True)
    mode = _messages.EnumField('ModeValueValuesEnum', 2)