from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FleetObservabilityBaseFeatureState(_messages.Message):
    """Base state for fleet observability feature.

  Enums:
    CodeValueValuesEnum: The high-level, machine-readable status of this
      Feature.

  Fields:
    code: The high-level, machine-readable status of this Feature.
    errors: Errors after reconciling the monitoring and logging feature if the
      code is not OK.
  """

    class CodeValueValuesEnum(_messages.Enum):
        """The high-level, machine-readable status of this Feature.

    Values:
      CODE_UNSPECIFIED: Unknown or not set.
      OK: The Feature is operating normally.
      ERROR: The Feature is encountering errors in the reconciliation. The
        Feature may need intervention to return to normal operation. See the
        description and any associated Feature-specific details for more
        information.
    """
        CODE_UNSPECIFIED = 0
        OK = 1
        ERROR = 2
    code = _messages.EnumField('CodeValueValuesEnum', 1)
    errors = _messages.MessageField('FeatureError', 2, repeated=True)