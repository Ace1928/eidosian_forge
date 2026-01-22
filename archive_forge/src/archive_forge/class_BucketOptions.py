from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BucketOptions(_messages.Message):
    """`BucketOptions` describes the bucket boundaries used in the histogram.

  Fields:
    exponential: Bucket boundaries grow exponentially.
    linear: Bucket boundaries grow linearly.
  """
    exponential = _messages.MessageField('Base2Exponent', 1)
    linear = _messages.MessageField('Linear', 2)