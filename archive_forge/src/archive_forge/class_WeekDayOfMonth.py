from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WeekDayOfMonth(_messages.Message):
    """`WeekDayOfMonth` defines the week day of the month on which the backups
  will run. The message combines a `WeekOfMonth` and `DayOfWeek` to produce
  values like `FIRST`/`MONDAY` or `LAST`/`FRIDAY`.

  Enums:
    DayOfWeekValueValuesEnum: Required. Specifies the day of the week.
    WeekOfMonthValueValuesEnum: Required. Specifies the week of the month.

  Fields:
    dayOfWeek: Required. Specifies the day of the week.
    weekOfMonth: Required. Specifies the week of the month.
  """

    class DayOfWeekValueValuesEnum(_messages.Enum):
        """Required. Specifies the day of the week.

    Values:
      DAY_OF_WEEK_UNSPECIFIED: The day of the week is unspecified.
      MONDAY: Monday
      TUESDAY: Tuesday
      WEDNESDAY: Wednesday
      THURSDAY: Thursday
      FRIDAY: Friday
      SATURDAY: Saturday
      SUNDAY: Sunday
    """
        DAY_OF_WEEK_UNSPECIFIED = 0
        MONDAY = 1
        TUESDAY = 2
        WEDNESDAY = 3
        THURSDAY = 4
        FRIDAY = 5
        SATURDAY = 6
        SUNDAY = 7

    class WeekOfMonthValueValuesEnum(_messages.Enum):
        """Required. Specifies the week of the month.

    Values:
      WEEK_OF_MONTH_UNSPECIFIED: The zero value. Do not use.
      FIRST: The first week of the month.
      SECOND: The second week of the month.
      THIRD: The third week of the month.
      FOURTH: The fourth week of the month.
      LAST: The last week of the month.
    """
        WEEK_OF_MONTH_UNSPECIFIED = 0
        FIRST = 1
        SECOND = 2
        THIRD = 3
        FOURTH = 4
        LAST = 5
    dayOfWeek = _messages.EnumField('DayOfWeekValueValuesEnum', 1)
    weekOfMonth = _messages.EnumField('WeekOfMonthValueValuesEnum', 2)