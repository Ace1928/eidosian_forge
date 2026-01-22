from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ResultStateValueValuesEnum(_messages.Enum):
    """The result state of the ScanRun. This field is only available after
    the execution state reaches "FINISHED".

    Values:
      RESULT_STATE_UNSPECIFIED: Default value. This value is returned when the
        ScanRun is not yet finished.
      SUCCESS: The scan finished without errors.
      ERROR: The scan finished with errors.
      KILLED: The scan was terminated by user.
    """
    RESULT_STATE_UNSPECIFIED = 0
    SUCCESS = 1
    ERROR = 2
    KILLED = 3