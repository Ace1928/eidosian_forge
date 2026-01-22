from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TimestampOrderValueValuesEnum(_messages.Enum):
    """Order the sequences in increasing or decreasing order of timestamps.
    Default is descending order of timestamps (latest first).

    Values:
      ORDER_UNSPECIFIED: Unspecified order. Default is Descending.
      ASCENDING: Ascending sort order.
      DESCENDING: Descending sort order.
    """
    ORDER_UNSPECIFIED = 0
    ASCENDING = 1
    DESCENDING = 2