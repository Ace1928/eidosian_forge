from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiServicecontrolV1Distribution(_messages.Message):
    """Distribution represents a frequency distribution of double-valued sample
  points. It contains the size of the population of sample points plus
  additional optional information: * the arithmetic mean of the samples * the
  minimum and maximum of the samples * the sum-squared-deviation of the
  samples, used to compute variance * a histogram of the values of the sample
  points

  Fields:
    bucketCounts: The number of samples in each histogram bucket.
      `bucket_counts` are optional. If present, they must sum to the `count`
      value. The buckets are defined below in `bucket_option`. There are N
      buckets. `bucket_counts[0]` is the number of samples in the underflow
      bucket. `bucket_counts[1]` to `bucket_counts[N-1]` are the numbers of
      samples in each of the finite buckets. And `bucket_counts[N] is the
      number of samples in the overflow bucket. See the comments of
      `bucket_option` below for more details. Any suffix of trailing zeros may
      be omitted.
    count: The total number of samples in the distribution. Must be >= 0.
    exemplars: Example points. Must be in increasing order of `value` field.
    explicitBuckets: Buckets with arbitrary user-provided width.
    exponentialBuckets: Buckets with exponentially growing width.
    linearBuckets: Buckets with constant width.
    maximum: The maximum of the population of values. Ignored if `count` is
      zero.
    mean: The arithmetic mean of the samples in the distribution. If `count`
      is zero then this field must be zero.
    minimum: The minimum of the population of values. Ignored if `count` is
      zero.
    sumOfSquaredDeviation: The sum of squared deviations from the mean:
      Sum[i=1..count]((x_i - mean)^2) where each x_i is a sample values. If
      `count` is zero then this field must be zero, otherwise validation of
      the request fails.
  """
    bucketCounts = _messages.IntegerField(1, repeated=True)
    count = _messages.IntegerField(2)
    exemplars = _messages.MessageField('Exemplar', 3, repeated=True)
    explicitBuckets = _messages.MessageField('GoogleApiServicecontrolV1ExplicitBuckets', 4)
    exponentialBuckets = _messages.MessageField('GoogleApiServicecontrolV1ExponentialBuckets', 5)
    linearBuckets = _messages.MessageField('GoogleApiServicecontrolV1LinearBuckets', 6)
    maximum = _messages.FloatField(7)
    mean = _messages.FloatField(8)
    minimum = _messages.FloatField(9)
    sumOfSquaredDeviation = _messages.FloatField(10)