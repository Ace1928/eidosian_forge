from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TaskStatusValueValuesEnum(_messages.Enum):
    """Optional. List only tasks in the state.

    Values:
      TASK_STATUS_UNSPECIFIED: <no description>
      TASK_STATUS_RUNNING: <no description>
      TASK_STATUS_SUCCESS: <no description>
      TASK_STATUS_FAILED: <no description>
      TASK_STATUS_KILLED: <no description>
      TASK_STATUS_PENDING: <no description>
    """
    TASK_STATUS_UNSPECIFIED = 0
    TASK_STATUS_RUNNING = 1
    TASK_STATUS_SUCCESS = 2
    TASK_STATUS_FAILED = 3
    TASK_STATUS_KILLED = 4
    TASK_STATUS_PENDING = 5