from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CalendarPeriodValueValuesEnum(_messages.Enum):
    """Optional. A CalendarPeriod represents the abstract concept of a time
    period that has a canonical start.

    Values:
      CALENDAR_PERIOD_UNSPECIFIED: Unspecified.
      MONTH: The month starts on the first date of the month and resets at the
        beginning of each month.
      QUARTER: The quarter starts on dates January 1, April 1, July 1, and
        October 1 of each year and resets at the beginning of the next
        quarter.
      YEAR: The year starts on January 1 and resets at the beginning of the
        next year.
      WEEK: The week period starts and resets every Monday.
      DAY: The day starts at 12:00am.
    """
    CALENDAR_PERIOD_UNSPECIFIED = 0
    MONTH = 1
    QUARTER = 2
    YEAR = 3
    WEEK = 4
    DAY = 5