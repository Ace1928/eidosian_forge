from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2KMapEstimationResult(_messages.Message):
    """Result of the reidentifiability analysis. Note that these results are an
  estimation, not exact values.

  Fields:
    kMapEstimationHistogram: The intervals [min_anonymity, max_anonymity] do
      not overlap. If a value doesn't correspond to any such interval, the
      associated frequency is zero. For example, the following records:
      {min_anonymity: 1, max_anonymity: 1, frequency: 17} {min_anonymity: 2,
      max_anonymity: 3, frequency: 42} {min_anonymity: 5, max_anonymity: 10,
      frequency: 99} mean that there are no record with an estimated anonymity
      of 4, 5, or larger than 10.
  """
    kMapEstimationHistogram = _messages.MessageField('GooglePrivacyDlpV2KMapEstimationHistogramBucket', 1, repeated=True)