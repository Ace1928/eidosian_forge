from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StandardSchedule(_messages.Message):
    """`StandardSchedule` defines a schedule that run within the confines of a
  defined window of days. We can define recurrence type for schedule as
  HOURLY, DAILY, WEEKLY, MONTHLY or YEARLY.

  Enums:
    DaysOfWeekValueListEntryValuesEnum:
    MonthsValueListEntryValuesEnum:
    RecurrenceTypeValueValuesEnum: Required. Specifies the `RecurrenceType`
      for the schedule.

  Fields:
    backupWindow: Required. A BackupWindow defines the window of day during
      which backup jobs will run. Jobs are queued at the beginning of the
      window and will be marked as `NOT_RUN` if they do not start by the end
      of the window. Note: running jobs will not be cancelled at the end of
      the window.
    daysOfMonth: Optional. Specifies days of months like 1, 5, or 14 on which
      jobs will run. Values for `days_of_month` are only applicable for
      `recurrence_type`, `MONTHLY` and `YEARLY`. A validation error will occur
      if other values are supplied.
    daysOfWeek: Optional. Specifies days of week like, MONDAY or TUESDAY, on
      which jobs will run. This is required for `recurrence_type`, `WEEKLY`
      and is not applicable otherwise. A validation error will occur if a
      value is supplied and `recurrence_type` is not `WEEKLY`.
    hourlyFrequency: Optional. Specifies frequency for hourly backups. A
      hourly frequency of 2 means jobs will run every 2 hours from start time
      till end time defined. This is required for `recurrence_type`, `HOURLY`
      and is not applicable otherwise. A validation error will occur if a
      value is supplied and `recurrence_type` is not `HOURLY`. Value of hourly
      frequency should be between 6 and 23. Reason for limit : We found that
      there is bandwidth limitation of 3GB/S for GMI while taking a backup and
      5GB/S while doing a restore. Given the amount of parallel backups and
      restore we are targeting, this will potentially take the backup time to
      mins and hours (in worst case scenario).
    months: Optional. Specifies the months of year, like `FEBRUARY` and/or
      `MAY`, on which jobs will run. This field is only applicable when
      `recurrence_type` is `YEARLY`. A validation error will occur if other
      values are supplied.
    recurrenceType: Required. Specifies the `RecurrenceType` for the schedule.
    repeatInterval: Required. TODO b/325560313: Deprecated and field will be
      removed after UI integration change. Repeat interval
    timeZone: Optional. The time zone to be used when interpreting the
      schedule. The value of this field must be a time zone name from the IANA
      tz database. See
      https://en.wikipedia.org/wiki/List_of_tz_database_time_zones for the
      list of valid timezone names. The default value is UTC. For e.g.,
      Europe/Paris.
    weekDayOfMonth: Optional. This will be supported in GA not in PP.
      Specifies a week day of the month like, FIRST SUNDAY or LAST MONDAY, on
      which jobs will run. This will be specified by two fields in
      `WeekDayOfMonth`, one for the day, e.g. `MONDAY`, and one for the week,
      e.g. `LAST`. This field is only applicable for `recurrence_type`,
      `MONTHLY` and `YEARLY`. A validation error will occur if other values
      are supplied.
  """

    class DaysOfWeekValueListEntryValuesEnum(_messages.Enum):
        """DaysOfWeekValueListEntryValuesEnum enum type.

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

    class MonthsValueListEntryValuesEnum(_messages.Enum):
        """MonthsValueListEntryValuesEnum enum type.

    Values:
      MONTH_UNSPECIFIED: The unspecified month.
      JANUARY: The month of January.
      FEBRUARY: The month of February.
      MARCH: The month of March.
      APRIL: The month of April.
      MAY: The month of May.
      JUNE: The month of June.
      JULY: The month of July.
      AUGUST: The month of August.
      SEPTEMBER: The month of September.
      OCTOBER: The month of October.
      NOVEMBER: The month of November.
      DECEMBER: The month of December.
    """
        MONTH_UNSPECIFIED = 0
        JANUARY = 1
        FEBRUARY = 2
        MARCH = 3
        APRIL = 4
        MAY = 5
        JUNE = 6
        JULY = 7
        AUGUST = 8
        SEPTEMBER = 9
        OCTOBER = 10
        NOVEMBER = 11
        DECEMBER = 12

    class RecurrenceTypeValueValuesEnum(_messages.Enum):
        """Required. Specifies the `RecurrenceType` for the schedule.

    Values:
      RECURRENCE_TYPE_UNSPECIFIED: recurrence type not set
      HOURLY: The `BackupRule` is to be applied hourly.
      DAILY: The `BackupRule` is to be applied daily.
      WEEKLY: The `BackupRule` is to be applied weekly.
      MONTHLY: The `BackupRule` is to be applied monthly.
      YEARLY: The `BackupRule` is to be applied yearly.
    """
        RECURRENCE_TYPE_UNSPECIFIED = 0
        HOURLY = 1
        DAILY = 2
        WEEKLY = 3
        MONTHLY = 4
        YEARLY = 5
    backupWindow = _messages.MessageField('BackupWindow', 1)
    daysOfMonth = _messages.IntegerField(2, repeated=True, variant=_messages.Variant.INT32)
    daysOfWeek = _messages.EnumField('DaysOfWeekValueListEntryValuesEnum', 3, repeated=True)
    hourlyFrequency = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    months = _messages.EnumField('MonthsValueListEntryValuesEnum', 5, repeated=True)
    recurrenceType = _messages.EnumField('RecurrenceTypeValueValuesEnum', 6)
    repeatInterval = _messages.IntegerField(7)
    timeZone = _messages.StringField(8)
    weekDayOfMonth = _messages.MessageField('WeekDayOfMonth', 9)