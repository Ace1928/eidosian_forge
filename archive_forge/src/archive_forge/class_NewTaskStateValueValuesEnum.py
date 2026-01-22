from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NewTaskStateValueValuesEnum(_messages.Enum):
    """The new task state.

    Values:
      STATE_UNSPECIFIED: Unknown state.
      PENDING: The Task is created and waiting for resources.
      ASSIGNED: The Task is assigned to at least one VM.
      RUNNING: The Task is running.
      FAILED: The Task has failed.
      SUCCEEDED: The Task has succeeded.
      UNEXECUTED: The Task has not been executed when the Job finishes.
    """
    STATE_UNSPECIFIED = 0
    PENDING = 1
    ASSIGNED = 2
    RUNNING = 3
    FAILED = 4
    SUCCEEDED = 5
    UNEXECUTED = 6