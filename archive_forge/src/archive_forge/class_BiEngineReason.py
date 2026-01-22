from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BiEngineReason(_messages.Message):
    """Reason why BI Engine didn't accelerate the query (or sub-query).

  Enums:
    CodeValueValuesEnum: Output only. High-level BI Engine reason for partial
      or disabled acceleration

  Fields:
    code: Output only. High-level BI Engine reason for partial or disabled
      acceleration
    message: Output only. Free form human-readable reason for partial or
      disabled acceleration.
  """

    class CodeValueValuesEnum(_messages.Enum):
        """Output only. High-level BI Engine reason for partial or disabled
    acceleration

    Values:
      CODE_UNSPECIFIED: BiEngineReason not specified.
      NO_RESERVATION: No reservation available for BI Engine acceleration.
      INSUFFICIENT_RESERVATION: Not enough memory available for BI Engine
        acceleration.
      UNSUPPORTED_SQL_TEXT: This particular SQL text is not supported for
        acceleration by BI Engine.
      INPUT_TOO_LARGE: Input too large for acceleration by BI Engine.
      OTHER_REASON: Catch-all code for all other cases for partial or disabled
        acceleration.
      TABLE_EXCLUDED: One or more tables were not eligible for BI Engine
        acceleration.
    """
        CODE_UNSPECIFIED = 0
        NO_RESERVATION = 1
        INSUFFICIENT_RESERVATION = 2
        UNSUPPORTED_SQL_TEXT = 3
        INPUT_TOO_LARGE = 4
        OTHER_REASON = 5
        TABLE_EXCLUDED = 6
    code = _messages.EnumField('CodeValueValuesEnum', 1)
    message = _messages.StringField(2)