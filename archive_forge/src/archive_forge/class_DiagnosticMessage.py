from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DiagnosticMessage(_messages.Message):
    """A message representing the key visualizer diagnostic messages.

  Enums:
    SeverityValueValuesEnum: The severity of the diagnostic message.

  Fields:
    info: Information about this diagnostic information.
    metric: The metric.
    metricSpecific: Whether this message is specific only for the current
      metric. By default Diagnostics are shown for all metrics, regardless
      which metric is the currently selected metric in the UI. However
      occasionally a metric will generate so many messages that the resulting
      visual clutter becomes overwhelming. In this case setting this to true,
      will show the diagnostic messages for that metric only if it is the
      currently selected metric.
    severity: The severity of the diagnostic message.
    shortMessage: The short message.
  """

    class SeverityValueValuesEnum(_messages.Enum):
        """The severity of the diagnostic message.

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
    info = _messages.MessageField('LocalizedString', 1)
    metric = _messages.MessageField('LocalizedString', 2)
    metricSpecific = _messages.BooleanField(3)
    severity = _messages.EnumField('SeverityValueValuesEnum', 4)
    shortMessage = _messages.MessageField('LocalizedString', 5)