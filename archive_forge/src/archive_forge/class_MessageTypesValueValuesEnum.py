from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MessageTypesValueValuesEnum(_messages.Enum):
    """Message types to return. If not populated - INFO, WARNING and ERROR
    messages are returned.

    Values:
      MESSAGE_SEVERITY_UNSPECIFIED: No severity specified.
      INFO: Informational message.
      WARNING: Warning message.
      ERROR: Error message.
    """
    MESSAGE_SEVERITY_UNSPECIFIED = 0
    INFO = 1
    WARNING = 2
    ERROR = 3