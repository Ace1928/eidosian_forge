from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
def HasCacheKeyPolicyArgs(args):
    """Returns true if the request requires a CacheKeyPolicy message.

  Args:
    args: The arguments passed to the gcloud command.

  Returns:
    True if there are cache key policy related arguments which require adding
    a CacheKeyPolicy message in the request.
  """
    return args.IsSpecified('cache_key_query_string_whitelist') or args.IsSpecified('cache_key_include_http_header')