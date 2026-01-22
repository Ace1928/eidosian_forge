from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AggregationsValueListEntryValuesEnum(_messages.Enum):
    """AggregationsValueListEntryValuesEnum enum type.

    Values:
      AGGREGATION_UNSPECIFIED: Unspecified.
      HOURLY: Insight should be aggregated at hourly level.
      DAILY: Insight should be aggregated at daily level.
      WEEKLY: Insight should be aggregated at weekly level.
      MONTHLY: Insight should be aggregated at monthly level.
      CUSTOM_DATE_RANGE: Insight should be aggregated at the custom date range
        passed in as the start and end time in the request.
    """
    AGGREGATION_UNSPECIFIED = 0
    HOURLY = 1
    DAILY = 2
    WEEKLY = 3
    MONTHLY = 4
    CUSTOM_DATE_RANGE = 5