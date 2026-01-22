from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataFrequencyValueValuesEnum(_messages.Enum):
    """The data frequency of a time series.

    Values:
      DATA_FREQUENCY_UNSPECIFIED: Default value.
      AUTO_FREQUENCY: Automatically inferred from timestamps.
      YEARLY: Yearly data.
      QUARTERLY: Quarterly data.
      MONTHLY: Monthly data.
      WEEKLY: Weekly data.
      DAILY: Daily data.
      HOURLY: Hourly data.
      PER_MINUTE: Per-minute data.
    """
    DATA_FREQUENCY_UNSPECIFIED = 0
    AUTO_FREQUENCY = 1
    YEARLY = 2
    QUARTERLY = 3
    MONTHLY = 4
    WEEKLY = 5
    DAILY = 6
    HOURLY = 7
    PER_MINUTE = 8