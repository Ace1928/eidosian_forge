from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExclusionWindow(_messages.Message):
    """Defines a time window during which no backup should happen. All time and
  date are in UTC.

  Fields:
    daily: The exclusion window occurs every day if set to "True". Specifying
      this field to "False" is an error.
    daysOfWeek: The exclusion window occurs on these days of each week in UTC.
    duration: Required. Specifies duration of the window. Duration must be >=
      5 minutes and < (target RPO - 20 minutes). Additional restrictions based
      on the recurrence type to allow some time for backup to happen: -
      single_occurrence_date: no restriction, but UI may warn about this when
      duration >= target RPO - daily window: duration < 24 hours - weekly
      window: - days of week includes all seven days of a week: duration < 24
      hours - all other weekly window: duration < 168 hours (i.e., 24 * 7
      hours)
    singleOccurrenceDate: No recurrence. The exclusion window occurs only once
      and on this date in UTC.
    startTime: Required. Specifies the start time of the window using time of
      the day in UTC.
  """
    daily = _messages.BooleanField(1)
    daysOfWeek = _messages.MessageField('DayOfWeekList', 2)
    duration = _messages.StringField(3)
    singleOccurrenceDate = _messages.MessageField('Date', 4)
    startTime = _messages.MessageField('TimeOfDay', 5)