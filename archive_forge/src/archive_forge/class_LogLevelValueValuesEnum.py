from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class LogLevelValueValuesEnum(_messages.Enum):
    """Indicates the severity of the log. Only relevant when action is `LOG`.

    Values:
      INFO: Information log message.
      WARNING: Warning log message.
      ERROR: Error log message.
    """
    INFO = 0
    WARNING = 1
    ERROR = 2