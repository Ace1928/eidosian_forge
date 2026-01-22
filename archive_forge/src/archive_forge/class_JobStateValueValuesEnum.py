from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JobStateValueValuesEnum(_messages.Enum):
    """Output only. The current state of the job. Default value is
    UNSPECIFIED

    Values:
      STATE_UNSPECIFIED: Job state is STATE_UNSPECIFIED for validate job
      QUEUED: job is queued
      IN_PROGRESS: job is in progress
      COMPLETED: job is completed
      ERRORED: job is errored
      STOP_IN_PROGRESS: Job stop in progress
      STOPPED: Job stopped
    """
    STATE_UNSPECIFIED = 0
    QUEUED = 1
    IN_PROGRESS = 2
    COMPLETED = 3
    ERRORED = 4
    STOP_IN_PROGRESS = 5
    STOPPED = 6