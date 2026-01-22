from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MaintenanceWindow(_messages.Message):
    """The configuration settings for Cloud Composer maintenance window. The
  following example: ``` { "startTime":"2019-08-01T01:00:00Z"
  "endTime":"2019-08-01T07:00:00Z" "recurrence":"FREQ=WEEKLY;BYDAY=TU,WE" }
  ``` would define a maintenance window between 01 and 07 hours UTC during
  each Tuesday and Wednesday.

  Fields:
    endTime: Required. Maintenance window end time. It is used only to
      calculate the duration of the maintenance window. The value for end_time
      must be in the future, relative to `start_time`.
    recurrence: Required. Maintenance window recurrence. Format is a subset of
      [RFC-5545](https://tools.ietf.org/html/rfc5545) `RRULE`. The only
      allowed values for `FREQ` field are `FREQ=DAILY` and
      `FREQ=WEEKLY;BYDAY=...` Example values: `FREQ=WEEKLY;BYDAY=TU,WE`,
      `FREQ=DAILY`.
    startTime: Required. Start time of the first recurrence of the maintenance
      window.
  """
    endTime = _messages.StringField(1)
    recurrence = _messages.StringField(2)
    startTime = _messages.StringField(3)