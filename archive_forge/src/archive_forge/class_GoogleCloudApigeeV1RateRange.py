from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1RateRange(_messages.Message):
    """API call volume range and the fees charged when the total number of API
  calls is within the range.

  Fields:
    end: Ending value of the range. Set to 0 or `null` for the last range of
      values.
    fee: Fee to charge when total number of API calls falls within this range.
    start: Starting value of the range. Set to 0 or `null` for the initial
      range of values.
  """
    end = _messages.IntegerField(1)
    fee = _messages.MessageField('GoogleTypeMoney', 2)
    start = _messages.IntegerField(3)