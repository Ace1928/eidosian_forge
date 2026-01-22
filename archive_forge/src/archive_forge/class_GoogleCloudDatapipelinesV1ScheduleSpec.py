from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatapipelinesV1ScheduleSpec(_messages.Message):
    """Details of the schedule the pipeline runs on.

  Fields:
    nextJobTime: Output only. When the next Scheduler job is going to run.
    schedule: Unix-cron format of the schedule. This information is retrieved
      from the linked Cloud Scheduler.
    timeZone: Timezone ID. This matches the timezone IDs used by the Cloud
      Scheduler API. If empty, UTC time is assumed.
  """
    nextJobTime = _messages.StringField(1)
    schedule = _messages.StringField(2)
    timeZone = _messages.StringField(3)