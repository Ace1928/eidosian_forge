from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ExecutionStateValueValuesEnum(_messages.Enum):
    """The execution state of the ScanRun.

    Values:
      EXECUTION_STATE_UNSPECIFIED: Represents an invalid state caused by
        internal server error. This value should never be returned.
      QUEUED: The scan is waiting in the queue.
      SCANNING: The scan is in progress.
      FINISHED: The scan is either finished or stopped by user.
    """
    EXECUTION_STATE_UNSPECIFIED = 0
    QUEUED = 1
    SCANNING = 2
    FINISHED = 3