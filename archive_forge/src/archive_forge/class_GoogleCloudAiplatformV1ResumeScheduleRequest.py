from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ResumeScheduleRequest(_messages.Message):
    """Request message for ScheduleService.ResumeSchedule.

  Fields:
    catchUp: Optional. Whether to backfill missed runs when the schedule is
      resumed from PAUSED state. If set to true, all missed runs will be
      scheduled. New runs will be scheduled after the backfill is complete.
      This will also update Schedule.catch_up field. Default to false.
  """
    catchUp = _messages.BooleanField(1)