from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2Value(_messages.Message):
    """Set of primitive values supported by the system. Note that for the
  purposes of inspection or transformation, the number of bytes considered to
  comprise a 'Value' is based on its representation as a UTF-8 encoded string.
  For example, if 'integer_value' is set to 123456789, the number of bytes
  would be counted as 9, even though an int64 only holds up to 8 bytes of
  data.

  Enums:
    DayOfWeekValueValueValuesEnum: day of week

  Fields:
    booleanValue: boolean
    dateValue: date
    dayOfWeekValue: day of week
    floatValue: float
    integerValue: integer
    stringValue: string
    timeValue: time of day
    timestampValue: timestamp
  """

    class DayOfWeekValueValueValuesEnum(_messages.Enum):
        """day of week

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
    booleanValue = _messages.BooleanField(1)
    dateValue = _messages.MessageField('GoogleTypeDate', 2)
    dayOfWeekValue = _messages.EnumField('DayOfWeekValueValueValuesEnum', 3)
    floatValue = _messages.FloatField(4)
    integerValue = _messages.IntegerField(5)
    stringValue = _messages.StringField(6)
    timeValue = _messages.MessageField('GoogleTypeTimeOfDay', 7)
    timestampValue = _messages.StringField(8)