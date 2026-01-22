from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class RecurringSchedule(_messages.Message):
    """Sets the time for recurring patch deployments.

  Enums:
    FrequencyValueValuesEnum: Required. The frequency unit of this recurring
      schedule.

  Fields:
    endTime: Optional. The end time at which a recurring patch deployment
      schedule is no longer active.
    frequency: Required. The frequency unit of this recurring schedule.
    lastExecuteTime: Output only. The time the last patch job ran
      successfully.
    monthly: Required. Schedule with monthly executions.
    nextExecuteTime: Output only. The time the next patch job is scheduled to
      run.
    startTime: Optional. The time that the recurring schedule becomes
      effective. Defaults to `create_time` of the patch deployment.
    timeOfDay: Required. Time of the day to run a recurring deployment.
    timeZone: Required. Defines the time zone that `time_of_day` is relative
      to. The rules for daylight saving time are determined by the chosen time
      zone.
    weekly: Required. Schedule with weekly executions.
  """

    class FrequencyValueValuesEnum(_messages.Enum):
        """Required. The frequency unit of this recurring schedule.

    Values:
      FREQUENCY_UNSPECIFIED: Invalid. A frequency must be specified.
      WEEKLY: Indicates that the frequency of recurrence should be expressed
        in terms of weeks.
      MONTHLY: Indicates that the frequency of recurrence should be expressed
        in terms of months.
      DAILY: Indicates that the frequency of recurrence should be expressed in
        terms of days.
    """
        FREQUENCY_UNSPECIFIED = 0
        WEEKLY = 1
        MONTHLY = 2
        DAILY = 3
    endTime = _messages.StringField(1)
    frequency = _messages.EnumField('FrequencyValueValuesEnum', 2)
    lastExecuteTime = _messages.StringField(3)
    monthly = _messages.MessageField('MonthlySchedule', 4)
    nextExecuteTime = _messages.StringField(5)
    startTime = _messages.StringField(6)
    timeOfDay = _messages.MessageField('TimeOfDay', 7)
    timeZone = _messages.MessageField('TimeZone', 8)
    weekly = _messages.MessageField('WeeklySchedule', 9)