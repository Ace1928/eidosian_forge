from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContextValue(_messages.Message):
    """A message representing context for a KeyRangeInfo, including a label,
  value, unit, and severity.

  Enums:
    SeverityValueValuesEnum: The severity of this context.

  Fields:
    label: The label for the context value. e.g. "latency".
    severity: The severity of this context.
    unit: The unit of the context value.
    value: The value for the context.
  """

    class SeverityValueValuesEnum(_messages.Enum):
        """The severity of this context.

    Values:
      SEVERITY_UNSPECIFIED: Required default value.
      INFO: Lowest severity level "Info".
      WARNING: Middle severity level "Warning".
      ERROR: Severity level signaling an error "Error"
      FATAL: Severity level signaling a non recoverable error "Fatal"
    """
        SEVERITY_UNSPECIFIED = 0
        INFO = 1
        WARNING = 2
        ERROR = 3
        FATAL = 4
    label = _messages.MessageField('LocalizedString', 1)
    severity = _messages.EnumField('SeverityValueValuesEnum', 2)
    unit = _messages.StringField(3)
    value = _messages.FloatField(4, variant=_messages.Variant.FLOAT)