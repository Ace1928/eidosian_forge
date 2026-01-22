from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2NumericalStatsResult(_messages.Message):
    """Result of the numerical stats computation.

  Fields:
    maxValue: Maximum value appearing in the column.
    minValue: Minimum value appearing in the column.
    quantileValues: List of 99 values that partition the set of field values
      into 100 equal sized buckets.
  """
    maxValue = _messages.MessageField('GooglePrivacyDlpV2Value', 1)
    minValue = _messages.MessageField('GooglePrivacyDlpV2Value', 2)
    quantileValues = _messages.MessageField('GooglePrivacyDlpV2Value', 3, repeated=True)