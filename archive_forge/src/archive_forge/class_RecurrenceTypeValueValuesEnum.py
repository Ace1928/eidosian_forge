from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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