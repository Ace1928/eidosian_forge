from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2KAnonymityResult(_messages.Message):
    """Result of the k-anonymity computation.

  Fields:
    equivalenceClassHistogramBuckets: Histogram of k-anonymity equivalence
      classes.
  """
    equivalenceClassHistogramBuckets = _messages.MessageField('GooglePrivacyDlpV2KAnonymityHistogramBucket', 1, repeated=True)