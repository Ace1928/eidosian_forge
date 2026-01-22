from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ScheduleRunResponse(_messages.Message):
    """Status of a scheduled run.

  Fields:
    runResponse: The response of the scheduled run.
    scheduledRunTime: The scheduled run time based on the user-specified
      schedule.
  """
    runResponse = _messages.StringField(1)
    scheduledRunTime = _messages.StringField(2)