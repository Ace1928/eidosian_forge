from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ScheduleStateValueValuesEnum(_messages.Enum):
    """Output only. Schedule state when the monitoring job is in Running
    state.

    Values:
      MONITORING_SCHEDULE_STATE_UNSPECIFIED: Unspecified state.
      PENDING: The pipeline is picked up and wait to run.
      OFFLINE: The pipeline is offline and will be scheduled for next run.
      RUNNING: The pipeline is running.
    """
    MONITORING_SCHEDULE_STATE_UNSPECIFIED = 0
    PENDING = 1
    OFFLINE = 2
    RUNNING = 3