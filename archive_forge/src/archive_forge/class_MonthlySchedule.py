from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class MonthlySchedule(_messages.Message):
    """Represents a monthly schedule. An example of a valid monthly schedule is
  "on the third Tuesday of the month" or "on the 15th of the month".

  Fields:
    monthDay: Required. One day of the month. 1-31 indicates the 1st to the
      31st day. -1 indicates the last day of the month. Months without the
      target day will be skipped. For example, a schedule to run "every month
      on the 31st" will not run in February, April, June, etc.
    weekDayOfMonth: Required. Week day in a month.
  """
    monthDay = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    weekDayOfMonth = _messages.MessageField('WeekDayOfMonth', 2)