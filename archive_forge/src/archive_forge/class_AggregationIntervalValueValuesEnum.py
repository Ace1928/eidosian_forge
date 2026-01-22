from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AggregationIntervalValueValuesEnum(_messages.Enum):
    """Optional. The aggregation interval for the logs. Default value is
    INTERVAL_5_SEC.

    Values:
      AGGREGATION_INTERVAL_UNSPECIFIED: If not specified, will default to
        INTERVAL_5_SEC.
      INTERVAL_5_SEC: Aggregate logs in 5s intervals.
      INTERVAL_30_SEC: Aggregate logs in 30s intervals.
      INTERVAL_1_MIN: Aggregate logs in 1m intervals.
      INTERVAL_5_MIN: Aggregate logs in 5m intervals.
      INTERVAL_10_MIN: Aggregate logs in 10m intervals.
      INTERVAL_15_MIN: Aggregate logs in 15m intervals.
    """
    AGGREGATION_INTERVAL_UNSPECIFIED = 0
    INTERVAL_5_SEC = 1
    INTERVAL_30_SEC = 2
    INTERVAL_1_MIN = 3
    INTERVAL_5_MIN = 4
    INTERVAL_10_MIN = 5
    INTERVAL_15_MIN = 6