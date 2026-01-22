from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2CategoricalStatsResult(_messages.Message):
    """Result of the categorical stats computation.

  Fields:
    valueFrequencyHistogramBuckets: Histogram of value frequencies in the
      column.
  """
    valueFrequencyHistogramBuckets = _messages.MessageField('GooglePrivacyDlpV2CategoricalStatsHistogramBucket', 1, repeated=True)