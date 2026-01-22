from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotificationLeadtimeValueValuesEnum(_messages.Enum):
    """Optional. Notification scheduling lead time.

    Values:
      NOTIFICATION_LEAD_TIME_UNSPECIFIED: Not set.
      WEEK1: WEEK1 == EARLIER with minimum 7d advanced notification. {7d, 14d}
      WEEK2: WEEK2 == LATER with minimum 14d advanced notification {14d, 21d}.
      WEEK5: WEEK5 == 40d support. minimum 35d advanced notification {35d,
        42d}.
    """
    NOTIFICATION_LEAD_TIME_UNSPECIFIED = 0
    WEEK1 = 1
    WEEK2 = 2
    WEEK5 = 3