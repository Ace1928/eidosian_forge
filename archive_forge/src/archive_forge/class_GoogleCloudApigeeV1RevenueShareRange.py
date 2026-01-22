from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1RevenueShareRange(_messages.Message):
    """API call volume range and the percentage of revenue to share with the
  developer when the total number of API calls is within the range.

  Fields:
    end: Ending value of the range. Set to 0 or `null` for the last range of
      values.
    sharePercentage: Percentage of the revenue to be shared with the
      developer. For example, to share 21 percent of the total revenue with
      the developer, set this value to 21. Specify a decimal number with a
      maximum of two digits following the decimal point.
    start: Starting value of the range. Set to 0 or `null` for the initial
      range of values.
  """
    end = _messages.IntegerField(1)
    sharePercentage = _messages.FloatField(2)
    start = _messages.IntegerField(3)