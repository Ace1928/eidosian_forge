from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TimeFrameValueValuesEnum(_messages.Enum):
    """Time frame of the report.

    Values:
      TIME_FRAME_UNSPECIFIED: The time frame was not specified and will
        default to WEEK.
      WEEK: One week.
      MONTH: One month.
      YEAR: One year.
    """
    TIME_FRAME_UNSPECIFIED = 0
    WEEK = 1
    MONTH = 2
    YEAR = 3