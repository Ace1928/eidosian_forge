from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
def GetCacheKeyPolicy(client, args, backend_bucket):
    """Returns the cache key policy.

  Args:
    client: The client used by gcloud.
    args: The arguments passed to the gcloud command.
    backend_bucket: The backend bucket object. If the backend bucket object
      contains a cache key policy already, it is used as the base to apply
      changes based on args.

  Returns:
    The cache key policy.
  """
    cache_key_policy = client.messages.BackendBucketCdnPolicyCacheKeyPolicy()
    if backend_bucket.cdnPolicy is not None and backend_bucket.cdnPolicy.cacheKeyPolicy is not None:
        cache_key_policy = backend_bucket.cdnPolicy.cacheKeyPolicy
    if args.cache_key_include_http_header is not None:
        cache_key_policy.includeHttpHeaders = args.cache_key_include_http_header
    if args.cache_key_query_string_whitelist is not None:
        cache_key_policy.queryStringWhitelist = args.cache_key_query_string_whitelist
    return cache_key_policy