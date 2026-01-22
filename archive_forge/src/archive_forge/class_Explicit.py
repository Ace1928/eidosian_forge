from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Explicit(_messages.Message):
    """Specifies a set of buckets with arbitrary widths.There are size(bounds)
  + 1 (= N) buckets. Bucket i has the following boundaries:Upper bound (0 <= i
  < N-1): boundsi Lower bound (1 <= i < N); boundsi - 1The bounds field must
  contain at least one element. If bounds has only one element, then there are
  no finite buckets, and that single element is the common boundary of the
  overflow and underflow buckets.

  Fields:
    bounds: The values must be monotonically increasing.
  """
    bounds = _messages.FloatField(1, repeated=True)