from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackendServiceCdnPolicyNegativeCachingPolicy(_messages.Message):
    """Specify CDN TTLs for response error codes.

  Fields:
    code: The HTTP status code to define a TTL against. Only HTTP status codes
      300, 301, 302, 307, 308, 404, 405, 410, 421, 451 and 501 are can be
      specified as values, and you cannot specify a status code more than
      once.
    ttl: The TTL (in seconds) for which to cache responses with the
      corresponding status code. The maximum allowed value is 1800s (30
      minutes), noting that infrequently accessed objects may be evicted from
      the cache before the defined TTL.
  """
    code = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    ttl = _messages.IntegerField(2, variant=_messages.Variant.INT32)