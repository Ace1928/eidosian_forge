from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiServicecontrolV1LinearBuckets(_messages.Message):
    """Describing buckets with constant width.

  Fields:
    numFiniteBuckets: The number of finite buckets. With the underflow and
      overflow buckets, the total number of buckets is `num_finite_buckets` +
      2. See comments on `bucket_options` for details.
    offset: The i'th linear bucket covers the interval [offset + (i-1) *
      width, offset + i * width) where i ranges from 1 to num_finite_buckets,
      inclusive.
    width: The i'th linear bucket covers the interval [offset + (i-1) * width,
      offset + i * width) where i ranges from 1 to num_finite_buckets,
      inclusive. Must be strictly positive.
  """
    numFiniteBuckets = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    offset = _messages.FloatField(2)
    width = _messages.FloatField(3)